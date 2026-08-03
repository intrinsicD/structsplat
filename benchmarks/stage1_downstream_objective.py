"""BENCH-019: bind Stage-1 fields to a fixed downstream objective without rewriting them.

The adapter in this module is deliberately passive.  Realtime-gs owns field loading, exact
additive/normalized queries, lifting, training, and held-out evaluation.  This module freezes the
cross-repository inputs and schedule, validates exported cell receipts, checks an A/A replay, and
analyzes field-family rankings with frame/capture-clustered uncertainty.

Lifecycle::

    python -m benchmarks.stage1_downstream_objective template --output protocol.draft.json
    python -m benchmarks.stage1_downstream_objective prepare-review \
        --draft protocol.draft.json --output protocol.review.json
    # A distinct reviewer writes the small review JSON described by ``review-template``.
    python -m benchmarks.stage1_downstream_objective finalize \
        --reviewed protocol.review.json --review review.json --output protocol.frozen.json
    python -m benchmarks.stage1_downstream_objective plan --protocol protocol.frozen.json
    python -m benchmarks.stage1_downstream_objective analyze \
        --protocol protocol.frozen.json --rows downstream_rows.jsonl --outdir report

No command launches realtime-gs.  The frozen manifest records its exact canonical command and
outcome root; the external repository retains execution authority and emits the rows consumed by
``analyze``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_SCHEMA = "structsplat.bench019.protocol.v1"
REVIEW_SCHEMA = "structsplat.bench019.protocol_review.v1"
ROW_SCHEMA = "structsplat.bench019.cell.v1"
REPORT_SCHEMA = "structsplat.bench019.report.v1"
RESULT_SCHEMA = ROW_SCHEMA
REQUIRED_CELL_ARTIFACTS = (
    "field",
    "history",
    "config",
    "target",
    "reconstruction",
    "error",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ProtocolError(ValueError):
    """Raised when a BENCH-019 protocol or result row fails closed."""


def canonical_json(value: object) -> bytes:
    """Return the one byte representation used by every BENCH-019 digest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _slug(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "item"


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ProtocolError(f"{label} must be a non-empty portable identifier")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
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


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError(f"{label} must be finite")
    return number


def _artifact_path(record: Mapping[str, Any], base: Path) -> Path:
    raw = _string(record.get("path"), "artifact.path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _seal_artifact(record: object, base: Path, label: str) -> dict[str, Any]:
    value = _mapping(record, label)
    path = _artifact_path(value, base)
    if not path.is_file():
        raise ProtocolError(f"{label} does not exist as a regular file: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _validate_artifact(record: object, base: Path, label: str) -> Path:
    value = _mapping(record, label)
    if set(value) != {"path", "sha256", "bytes"}:
        raise ProtocolError(f"{label} must contain exactly path, sha256, and bytes")
    path = _artifact_path(value, base)
    digest = value.get("sha256")
    size = value.get("bytes")
    if not isinstance(digest, str) or not HEX64.fullmatch(digest):
        raise ProtocolError(f"{label}.sha256 must be a lowercase SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ProtocolError(f"{label}.bytes must be a non-negative integer")
    if not path.is_file():
        raise ProtocolError(f"{label} is missing: {path}")
    if path.stat().st_size != size:
        raise ProtocolError(f"{label} byte count differs from its binding")
    if sha256_file(path) != digest:
        raise ProtocolError(f"{label} SHA-256 differs from its binding")
    return path


def _artifact_records(protocol: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for index, repository in enumerate(_list(protocol.get("repositories"), "repositories")):
        record = _mapping(repository, f"repositories[{index}]")
        yield f"repository_{record.get('name', index)}_environment", _mapping(
            record.get("environment"), f"repositories[{index}].environment"
        )
    for capture_index, capture in enumerate(_list(protocol.get("captures"), "captures")):
        capture_record = _mapping(capture, f"captures[{capture_index}]")
        for frame_index, frame in enumerate(
            _list(capture_record.get("frames"), f"captures[{capture_index}].frames")
        ):
            frame_record = _mapping(
                frame, f"captures[{capture_index}].frames[{frame_index}]"
            )
            prefix = f"{capture_record.get('id', capture_index)}_{frame_record.get('id', frame_index)}"
            for name in ("pixels", "masks", "cameras"):
                yield f"{prefix}_{name}", _mapping(
                    frame_record.get(name), f"{prefix}.{name}"
                )
            for family_index, family in enumerate(
                _list(frame_record.get("families"), f"{prefix}.families")
            ):
                family_record = _mapping(family, f"{prefix}.families[{family_index}]")
                family_prefix = f"{prefix}_{family_record.get('id', family_index)}"
                for name in ("field_manifest", "stage1_metrics"):
                    yield f"{family_prefix}_{name}", _mapping(
                        family_record.get(name), f"{family_prefix}.{name}"
                    )
    downstream = _mapping(protocol.get("downstream"), "downstream")
    for name in (
        "task_manifest",
        "dataset_manifest",
        "environment",
        "schedule_config",
    ):
        yield f"downstream_{name}", _mapping(downstream.get(name), f"downstream.{name}")
    review = protocol.get("review")
    if isinstance(review, Mapping) and isinstance(review.get("artifact"), Mapping):
        yield "prospective_review", _mapping(review["artifact"], "review.artifact")


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolError(f"cannot inspect Git repository {root}: {exc}") from exc


def _seal_repository(record: object, base: Path, index: int) -> dict[str, Any]:
    value = _mapping(record, f"repositories[{index}]")
    name = _identifier(value.get("name"), f"repositories[{index}].name")
    root_raw = _string(value.get("root"), f"repositories[{index}].root")
    root = Path(root_raw).expanduser()
    if not root.is_absolute():
        root = base / root
    root = root.resolve()
    status = _git(root, "status", "--short", "--untracked-files=normal")
    if status:
        raise ProtocolError(
            f"repository {name} is dirty; a formal protocol must bind a clean source commit"
        )
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    if not HEX40.fullmatch(commit) or not branch:
        raise ProtocolError(f"repository {name} has no usable commit/branch identity")
    return {
        "name": name,
        "root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": False,
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "environment": _seal_artifact(
            value.get("environment"), base, f"repositories[{index}].environment"
        ),
    }


def _validate_repository(record: object, base: Path, index: int) -> None:
    value = _mapping(record, f"repositories[{index}]")
    required = {
        "name",
        "root",
        "commit",
        "branch",
        "dirty",
        "status_sha256",
        "environment",
    }
    if set(value) != required:
        raise ProtocolError(f"repositories[{index}] has unexpected or missing fields")
    _identifier(value["name"], f"repositories[{index}].name")
    _string(value["root"], f"repositories[{index}].root")
    if not isinstance(value["commit"], str) or not HEX40.fullmatch(value["commit"]):
        raise ProtocolError(f"repositories[{index}].commit must be a Git SHA")
    _string(value["branch"], f"repositories[{index}].branch")
    if value["dirty"] is not False:
        raise ProtocolError(f"repositories[{index}] must be clean")
    if not isinstance(value["status_sha256"], str) or not HEX64.fullmatch(
        value["status_sha256"]
    ):
        raise ProtocolError(f"repositories[{index}].status_sha256 is invalid")
    if value["status_sha256"] != hashlib.sha256(b"").hexdigest():
        raise ProtocolError(f"repositories[{index}] does not carry the clean status digest")
    _validate_artifact(value["environment"], base, f"repositories[{index}].environment")


def _protocol_without_seals(protocol: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(protocol))
    for key in ("state", "design_sha256", "protocol_sha256", "review"):
        payload.pop(key, None)
    return payload


def design_digest(protocol: Mapping[str, Any]) -> str:
    """Digest the outcome-relevant design, excluding lifecycle/reviewer attestations."""
    return _digest(_protocol_without_seals(protocol))


def protocol_digest(protocol: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(protocol))
    payload.pop("protocol_sha256", None)
    return _digest(payload)


def _validate_semantics(value: object, label: str) -> None:
    record = _mapping(value, label)
    required = {
        "provider",
        "equation",
        "blend_mode",
        "alpha_policy",
        "coordinate_convention",
        "semantic_digest",
    }
    if set(record) != required:
        raise ProtocolError(f"{label} has unexpected or missing fields")
    _identifier(record["provider"], f"{label}.provider")
    equation = record["equation"]
    blend = record["blend_mode"]
    allowed = {
        "additive_sum": "additive",
        "normalized_weighted_sum": "normalized",
    }
    if equation not in allowed:
        raise ProtocolError(f"{label}.equation is not an exact supported equation")
    if blend != allowed[equation]:
        raise ProtocolError(f"{label} equation and blend_mode disagree")
    _identifier(record["alpha_policy"], f"{label}.alpha_policy")
    _string(record["coordinate_convention"], f"{label}.coordinate_convention")
    if not isinstance(record["semantic_digest"], str) or not HEX64.fullmatch(
        record["semantic_digest"]
    ):
        raise ProtocolError(f"{label}.semantic_digest must be a SHA-256")


def _metric_specs(protocol: Mapping[str, Any], kind: str) -> list[Mapping[str, Any]]:
    specs = []
    names: set[str] = set()
    for index, raw in enumerate(_list(protocol.get(kind), kind)):
        record = _mapping(raw, f"{kind}[{index}]")
        required = {"name", "direction"}
        if kind == "responses":
            required.add("primary")
        if set(record) != required:
            raise ProtocolError(f"{kind}[{index}] has unexpected or missing fields")
        name = _identifier(record["name"], f"{kind}[{index}].name")
        if name in names:
            raise ProtocolError(f"{kind} contains duplicate metric {name}")
        names.add(name)
        if record["direction"] not in {"higher", "lower"}:
            raise ProtocolError(f"{kind}[{index}].direction must be higher or lower")
        if kind == "responses" and not isinstance(record["primary"], bool):
            raise ProtocolError(f"{kind}[{index}].primary must be boolean")
        specs.append(record)
    if not specs:
        raise ProtocolError(f"{kind} must not be empty")
    if kind == "responses" and sum(bool(spec["primary"]) for spec in specs) != 1:
        raise ProtocolError("responses must declare exactly one primary metric")
    return specs


def _validate_analysis(protocol: Mapping[str, Any]) -> None:
    value = _mapping(protocol.get("analysis"), "analysis")
    required = {
        "bootstrap_replicates",
        "bootstrap_seed",
        "minimum_capture_groups",
        "minimum_frames",
        "minimum_family_count",
        "minimum_spearman",
        "minimum_bootstrap_lower",
        "minimum_lofo_top1_agreement",
        "selection_priority",
        "missing_policy",
    }
    if set(value) != required:
        raise ProtocolError("analysis has unexpected or missing fields")
    for name in (
        "bootstrap_replicates",
        "minimum_capture_groups",
        "minimum_frames",
        "minimum_family_count",
    ):
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ProtocolError(f"analysis.{name} must be a positive integer")
    if isinstance(value["bootstrap_seed"], bool) or not isinstance(
        value["bootstrap_seed"], int
    ):
        raise ProtocolError("analysis.bootstrap_seed must be an integer")
    for name in (
        "minimum_spearman",
        "minimum_bootstrap_lower",
    ):
        number = _finite_number(value[name], f"analysis.{name}")
        if not -1.0 <= number <= 1.0:
            raise ProtocolError(f"analysis.{name} must lie in [-1,1]")
    agreement = _finite_number(
        value["minimum_lofo_top1_agreement"],
        "analysis.minimum_lofo_top1_agreement",
    )
    if not 0.0 <= agreement <= 1.0:
        raise ProtocolError("analysis.minimum_lofo_top1_agreement must lie in [0,1]")
    predictors = {spec["name"] for spec in _metric_specs(protocol, "predictors")}
    priority = _list(value["selection_priority"], "analysis.selection_priority")
    if set(priority) != predictors or len(priority) != len(predictors):
        raise ProtocolError("analysis.selection_priority must list every predictor exactly once")
    if value["missing_policy"] != "fail_closed":
        raise ProtocolError("BENCH-019 supports only the preregistered fail_closed missing policy")


def _validate_aa(protocol: Mapping[str, Any], frame_families: Mapping[str, set[str]]) -> None:
    value = _mapping(protocol.get("aa_replay"), "aa_replay")
    required = {
        "frame_id",
        "family_id",
        "seed",
        "initializer",
        "primary_replicate",
        "replay_replicate",
        "metric_abs_tolerance",
    }
    if set(value) != required:
        raise ProtocolError("aa_replay has unexpected or missing fields")
    frame_id = _identifier(value["frame_id"], "aa_replay.frame_id")
    family_id = _identifier(value["family_id"], "aa_replay.family_id")
    if frame_id not in frame_families or family_id not in frame_families[frame_id]:
        raise ProtocolError("aa_replay references an unknown frame/family")
    downstream = _mapping(protocol["downstream"], "downstream")
    if value["seed"] not in downstream["seeds"]:
        raise ProtocolError("aa_replay.seed is not a frozen downstream seed")
    if value["initializer"] not in downstream["initializers"]:
        raise ProtocolError("aa_replay.initializer is not frozen")
    primary = _identifier(value["primary_replicate"], "aa_replay.primary_replicate")
    replay = _identifier(value["replay_replicate"], "aa_replay.replay_replicate")
    if primary == replay:
        raise ProtocolError("A/A replay labels must be distinct")
    tolerances = _mapping(value["metric_abs_tolerance"], "aa_replay.metric_abs_tolerance")
    known = {
        spec["name"]
        for kind in ("predictors", "responses")
        for spec in _metric_specs(protocol, kind)
    }
    primary_response = next(
        spec["name"] for spec in _metric_specs(protocol, "responses") if spec["primary"]
    )
    predictor_names = {spec["name"] for spec in _metric_specs(protocol, "predictors")}
    if (
        not tolerances
        or not set(tolerances) <= known
        or primary_response not in tolerances
        or not (set(tolerances) & predictor_names)
    ):
        raise ProtocolError(
            "A/A tolerances must be a known-metric subset containing the primary response "
            "and at least one Stage-1 predictor"
        )
    for name, tolerance in tolerances.items():
        if _finite_number(tolerance, f"aa_replay.metric_abs_tolerance.{name}") < 0.0:
            raise ProtocolError("A/A tolerances must be non-negative")


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    base: str | Path = ".",
    require_frozen: bool = False,
) -> None:
    """Validate a review-ready or frozen protocol and every bound artifact."""
    base_path = Path(base).resolve()
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ProtocolError(f"protocol schema must be {PROTOCOL_SCHEMA}")
    _identifier(protocol.get("task_id"), "task_id")
    _identifier(protocol.get("driver"), "driver")
    state = protocol.get("state")
    if state not in {"review", "frozen"}:
        raise ProtocolError("validated protocol state must be review or frozen")
    if require_frozen and state != "frozen":
        raise ProtocolError("formal analysis requires a frozen protocol")
    if protocol.get("claim_scope") not in {"general", "workload_specific"}:
        raise ProtocolError("claim_scope must be general or workload_specific")

    repositories = _list(protocol.get("repositories"), "repositories")
    if len(repositories) < 2:
        raise ProtocolError("BENCH-019 must bind both StructSplat and realtime-gs repositories")
    repository_names = []
    for index, repository in enumerate(repositories):
        _validate_repository(repository, base_path, index)
        repository_names.append(repository["name"])
    if len(set(repository_names)) != len(repository_names):
        raise ProtocolError("repository names must be unique")

    captures = _list(protocol.get("captures"), "captures")
    if not captures:
        raise ProtocolError("captures must not be empty")
    capture_ids: set[str] = set()
    frame_ids: set[str] = set()
    frame_families: dict[str, set[str]] = {}
    canonical_families: set[str] | None = None
    for capture_index, raw_capture in enumerate(captures):
        capture = _mapping(raw_capture, f"captures[{capture_index}]")
        if set(capture) != {"id", "frames"}:
            raise ProtocolError(f"captures[{capture_index}] has unexpected or missing fields")
        capture_id = _identifier(capture["id"], f"captures[{capture_index}].id")
        if capture_id in capture_ids:
            raise ProtocolError("capture IDs must be unique")
        capture_ids.add(capture_id)
        frames = _list(capture["frames"], f"captures[{capture_index}].frames")
        if not frames:
            raise ProtocolError(f"capture {capture_id} has no frames")
        for frame_index, raw_frame in enumerate(frames):
            frame = _mapping(raw_frame, f"{capture_id}.frames[{frame_index}]")
            required = {"id", "pixels", "masks", "cameras", "split", "families"}
            if set(frame) != required:
                raise ProtocolError(f"{capture_id}.frames[{frame_index}] has bad fields")
            frame_id = _identifier(frame["id"], f"{capture_id}.frames[{frame_index}].id")
            if frame_id in frame_ids:
                raise ProtocolError("frame IDs must be globally unique")
            frame_ids.add(frame_id)
            for name in ("pixels", "masks", "cameras"):
                _validate_artifact(frame[name], base_path, f"{capture_id}.{frame_id}.{name}")
            split = _mapping(frame["split"], f"{capture_id}.{frame_id}.split")
            if set(split) != {"train", "heldout"}:
                raise ProtocolError(f"{capture_id}.{frame_id}.split must contain train/heldout")
            train = [_identifier(v, "split train view") for v in _list(split["train"], "train")]
            heldout = [
                _identifier(v, "split heldout view") for v in _list(split["heldout"], "heldout")
            ]
            if not train or not heldout or set(train) & set(heldout):
                raise ProtocolError(f"{capture_id}.{frame_id} needs disjoint non-empty splits")
            if len(set(train)) != len(train) or len(set(heldout)) != len(heldout):
                raise ProtocolError(f"{capture_id}.{frame_id} split IDs must be unique")
            families = _list(frame["families"], f"{capture_id}.{frame_id}.families")
            family_ids: set[str] = set()
            for family_index, raw_family in enumerate(families):
                family = _mapping(raw_family, f"{capture_id}.{frame_id}.families[{family_index}]")
                if set(family) != {"id", "field_manifest", "stage1_metrics", "semantics"}:
                    raise ProtocolError(f"{capture_id}.{frame_id} family has bad fields")
                family_id = _identifier(family["id"], "family.id")
                if family_id in family_ids:
                    raise ProtocolError(f"{frame_id} has duplicate family {family_id}")
                family_ids.add(family_id)
                _validate_artifact(
                    family["field_manifest"], base_path, f"{frame_id}.{family_id}.field_manifest"
                )
                _validate_artifact(
                    family["stage1_metrics"], base_path, f"{frame_id}.{family_id}.stage1_metrics"
                )
                _validate_semantics(family["semantics"], f"{frame_id}.{family_id}.semantics")
            if canonical_families is None:
                canonical_families = family_ids
            elif family_ids != canonical_families:
                raise ProtocolError("every frame must contain the same paired field families")
            frame_families[frame_id] = family_ids

    downstream = _mapping(protocol.get("downstream"), "downstream")
    required_downstream = {
        "task_manifest",
        "dataset_manifest",
        "environment",
        "schedule_config",
        "command",
        "working_directory",
        "outcome_root",
        "seeds",
        "initializers",
        "result_schema",
    }
    if set(downstream) != required_downstream:
        raise ProtocolError("downstream has unexpected or missing fields")
    for name in ("task_manifest", "dataset_manifest", "environment", "schedule_config"):
        _validate_artifact(downstream[name], base_path, f"downstream.{name}")
    command = _list(downstream["command"], "downstream.command")
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ProtocolError("downstream.command must be a non-empty argv list")
    for name in ("working_directory", "outcome_root"):
        raw_path = Path(_string(downstream[name], f"downstream.{name}"))
        if not raw_path.is_absolute():
            raise ProtocolError(f"downstream.{name} must be absolute after preparation")
    if not Path(downstream["working_directory"]).is_dir():
        raise ProtocolError("downstream.working_directory must exist as a directory")
    seeds = _list(downstream["seeds"], "downstream.seeds")
    if (
        len(seeds) < 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ProtocolError("downstream.seeds must contain at least three unique integers")
    initializers = [
        _identifier(value, "downstream initializer")
        for value in _list(downstream["initializers"], "downstream.initializers")
    ]
    if not initializers or len(set(initializers)) != len(initializers):
        raise ProtocolError("downstream.initializers must be unique and non-empty")
    if downstream["result_schema"] != RESULT_SCHEMA:
        raise ProtocolError(f"downstream.result_schema must be {RESULT_SCHEMA}")

    predictor_names = {spec["name"] for spec in _metric_specs(protocol, "predictors")}
    response_names = {spec["name"] for spec in _metric_specs(protocol, "responses")}
    if predictor_names & response_names:
        raise ProtocolError("predictor and response metric names must be disjoint")
    _validate_analysis(protocol)
    if protocol["claim_scope"] == "general":
        if protocol["analysis"]["minimum_capture_groups"] < 3:
            raise ProtocolError("a general claim requires at least three capture groups")
        if protocol["analysis"]["minimum_frames"] < 2:
            raise ProtocolError("a general claim requires at least two frames")
    _validate_aa(protocol, frame_families)
    recorded_design = protocol.get("design_sha256")
    if not isinstance(recorded_design, str) or recorded_design != design_digest(protocol):
        raise ProtocolError("protocol design SHA-256 is missing or does not match")
    if state == "frozen":
        review = _mapping(protocol.get("review"), "review")
        required_review = {
            "driver",
            "reviewer",
            "verdict",
            "design_sha256",
            "artifact",
        }
        if set(review) != required_review:
            raise ProtocolError("frozen protocol review has unexpected or missing fields")
        driver = _identifier(review["driver"], "review.driver")
        reviewer = _identifier(review["reviewer"], "review.reviewer")
        if driver.casefold() == reviewer.casefold():
            raise ProtocolError("protocol reviewer must be distinct from the driver")
        if driver != protocol["driver"] or review["verdict"] != "approved":
            raise ProtocolError("frozen protocol does not carry an approval for its driver")
        if review["design_sha256"] != recorded_design:
            raise ProtocolError("review approved a different protocol design digest")
        _validate_artifact(review["artifact"], base_path, "review.artifact")
        recorded_protocol = protocol.get("protocol_sha256")
        if not isinstance(recorded_protocol, str) or recorded_protocol != protocol_digest(protocol):
            raise ProtocolError("frozen protocol SHA-256 is missing or does not match")


def _seal_protocol_artifacts(protocol: dict[str, Any], base: Path) -> None:
    protocol["repositories"] = [
        _seal_repository(record, base, index)
        for index, record in enumerate(_list(protocol.get("repositories"), "repositories"))
    ]
    for capture_index, capture in enumerate(protocol["captures"]):
        for frame_index, frame in enumerate(capture["frames"]):
            prefix = f"captures[{capture_index}].frames[{frame_index}]"
            for name in ("pixels", "masks", "cameras"):
                frame[name] = _seal_artifact(frame[name], base, f"{prefix}.{name}")
            for family_index, family in enumerate(frame["families"]):
                for name in ("field_manifest", "stage1_metrics"):
                    family[name] = _seal_artifact(
                        family[name], base, f"{prefix}.families[{family_index}].{name}"
                    )
    for name in ("task_manifest", "dataset_manifest", "environment", "schedule_config"):
        protocol["downstream"][name] = _seal_artifact(
            protocol["downstream"][name], base, f"downstream.{name}"
        )


def prepare_review(draft: Mapping[str, Any], *, base: str | Path = ".") -> dict[str, Any]:
    """Resolve a draft against clean repositories and immutable files for prospective review."""
    protocol = copy.deepcopy(dict(draft))
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("state") != "draft":
        raise ProtocolError("prepare-review requires a BENCH-019 draft protocol")
    protocol.pop("review", None)
    protocol.pop("design_sha256", None)
    protocol.pop("protocol_sha256", None)
    base_path = Path(base).resolve()
    _seal_protocol_artifacts(protocol, base_path)
    working = Path(
        _string(protocol["downstream"].get("working_directory"), "working_directory")
    ).expanduser()
    if not working.is_absolute():
        working = base_path / working
    working = working.resolve()
    if not working.is_dir():
        raise ProtocolError("downstream working directory does not exist")
    protocol["downstream"]["working_directory"] = str(working)
    outcome = Path(_string(protocol["downstream"]["outcome_root"], "outcome_root")).expanduser()
    if not outcome.is_absolute():
        outcome = base_path / outcome
    outcome = outcome.resolve()
    protocol["downstream"]["outcome_root"] = str(outcome)
    if outcome.exists() and (not outcome.is_dir() or any(outcome.iterdir())):
        raise ProtocolError("outcome root is already non-empty; prospective freezing is too late")
    protocol["state"] = "review"
    protocol["design_sha256"] = design_digest(protocol)
    validate_protocol(protocol, base=base_path)
    return protocol


def review_template(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("state") != "review":
        raise ProtocolError("review-template requires a review-ready protocol")
    validate_protocol(protocol)
    return {
        "schema": REVIEW_SCHEMA,
        "driver": protocol["driver"],
        "reviewer": "replace-with-distinct-reviewer",
        "verdict": "approved-or-rejected",
        "design_sha256": protocol["design_sha256"],
        "outcome_accessed": False,
        "notes": "Review exact inputs, splits, semantics, commands, metrics, and decision gates.",
    }


def finalize_protocol(
    reviewed: Mapping[str, Any],
    review_path: str | Path,
    *,
    base: str | Path = ".",
) -> dict[str, Any]:
    """Attach a distinct outcome-unseen review and freeze the exact protocol."""
    protocol = copy.deepcopy(dict(reviewed))
    base_path = Path(base).resolve()
    validate_protocol(protocol, base=base_path)
    if protocol.get("state") != "review":
        raise ProtocolError("finalize requires a review-ready protocol")
    review_file = Path(review_path).expanduser().resolve()
    review = json.loads(review_file.read_text(encoding="utf-8"))
    record = _mapping(review, "review file")
    required = {
        "schema",
        "driver",
        "reviewer",
        "verdict",
        "design_sha256",
        "outcome_accessed",
        "notes",
    }
    if set(record) != required or record["schema"] != REVIEW_SCHEMA:
        raise ProtocolError("review file has the wrong schema or fields")
    driver = _identifier(record["driver"], "review.driver")
    reviewer = _identifier(record["reviewer"], "review.reviewer")
    if driver != protocol["driver"] or driver.casefold() == reviewer.casefold():
        raise ProtocolError("reviewer must be distinct and the reviewed driver must match")
    if record["verdict"] != "approved" or record["outcome_accessed"] is not False:
        raise ProtocolError("formal protocol requires an approved, outcome-unseen review")
    if record["design_sha256"] != protocol["design_sha256"]:
        raise ProtocolError("review file approved a different design digest")
    outcome = Path(protocol["downstream"]["outcome_root"]).expanduser()
    if not outcome.is_absolute():
        outcome = base_path / outcome
    if outcome.exists() and (not outcome.is_dir() or any(outcome.iterdir())):
        raise ProtocolError("outcome root became non-empty before protocol finalization")
    protocol["review"] = {
        "driver": driver,
        "reviewer": reviewer,
        "verdict": "approved",
        "design_sha256": record["design_sha256"],
        "artifact": _seal_artifact({"path": str(review_file)}, base_path, "review artifact"),
    }
    protocol["state"] = "frozen"
    protocol["protocol_sha256"] = protocol_digest(protocol)
    validate_protocol(protocol, base=base_path, require_frozen=True)
    return protocol


def _frame_index(protocol: Mapping[str, Any]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    result = {}
    for capture in protocol["captures"]:
        for frame in capture["frames"]:
            result[frame["id"]] = (capture["id"], frame)
    return result


def expected_cell_keys(protocol: Mapping[str, Any]) -> list[tuple[str, str, int, str, str]]:
    """Return frame/family/seed/initializer/replicate cells in stable execution order."""
    primary = protocol["aa_replay"]["primary_replicate"]
    keys: list[tuple[str, str, int, str, str]] = []
    for capture in protocol["captures"]:
        for frame in capture["frames"]:
            for family in frame["families"]:
                for seed in protocol["downstream"]["seeds"]:
                    for initializer in protocol["downstream"]["initializers"]:
                        keys.append((frame["id"], family["id"], seed, initializer, primary))
    aa = protocol["aa_replay"]
    keys.append(
        (
            aa["frame_id"],
            aa["family_id"],
            aa["seed"],
            aa["initializer"],
            aa["replay_replicate"],
        )
    )
    return keys


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("rows"), list):
            return value["rows"]
        raise ProtocolError(f"{path} does not contain a result-row list")
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for name in ("stage1", "downstream", "artifacts"):
                if row.get(name):
                    row[name] = json.loads(row[name])
            if row.get("seed"):
                row["seed"] = int(row["seed"])
        return rows
    raise ProtocolError(f"unsupported row file extension: {path.suffix}")


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, int, str, str]:
    return (
        str(row.get("frame_id")),
        str(row.get("family_id")),
        int(row.get("seed", -1)),
        str(row.get("initializer")),
        str(row.get("replicate_id")),
    )


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int, str, str], Mapping[str, Any]]:
    indexed = {}
    for row in rows:
        if row.get("schema") != ROW_SCHEMA:
            continue
        try:
            key = _row_key(row)
        except (TypeError, ValueError):
            continue
        indexed[key] = row
    return indexed


def validate_result_rows(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Return all row/binding problems so a report can retain failures without hiding them."""
    problems: list[str] = []
    frames = _frame_index(protocol)
    expected = set(expected_cell_keys(protocol))
    predictor_names = [spec["name"] for spec in protocol["predictors"]]
    response_names = [spec["name"] for spec in protocol["responses"]]
    seen: set[tuple[str, str, int, str, str]] = set()
    stage1_values: dict[tuple[str, str], dict[str, float]] = {}
    factor_digests: dict[tuple[str, int, str, str], str] = {}
    required_row_fields = {
        "schema",
        "status",
        "error",
        "capture_id",
        "frame_id",
        "family_id",
        "seed",
        "initializer",
        "replicate_id",
        "field_manifest_sha256",
        "field_semantic_digest",
        "downstream_factor_digest",
        "stage1",
        "downstream",
        "artifacts",
    }
    for index, raw in enumerate(rows):
        label = f"rows[{index}]"
        row = dict(raw)
        if row.get("schema") != ROW_SCHEMA:
            problems.append(f"{label}: wrong schema")
            continue
        if set(row) != required_row_fields:
            problems.append(f"{label}: row has unexpected or missing fields")
        status = row.get("status")
        if status not in {"ok", "error"}:
            problems.append(f"{label}: status must be ok or error")
        if status == "error" and not str(row.get("error", "")).strip():
            problems.append(f"{label}: error row has no diagnostic")
        try:
            key = _row_key(row)
        except (TypeError, ValueError):
            problems.append(f"{label}: invalid stable cell key")
            continue
        if key in seen:
            problems.append(f"{label}: duplicate stable cell key {key}")
        seen.add(key)
        if key not in expected:
            problems.append(f"{label}: cell is not declared by the frozen protocol")
            continue
        frame_id, family_id, seed, initializer, replicate = key
        capture_id, frame = frames[frame_id]
        if row.get("capture_id") != capture_id:
            problems.append(f"{label}: capture_id does not match the frozen frame")
        family = next(item for item in frame["families"] if item["id"] == family_id)
        if row.get("field_manifest_sha256") != family["field_manifest"]["sha256"]:
            problems.append(f"{label}: field manifest binding differs")
        if row.get("field_semantic_digest") != family["semantics"]["semantic_digest"]:
            problems.append(f"{label}: field semantic digest differs")
        factor = row.get("downstream_factor_digest")
        if not isinstance(factor, str) or not HEX64.fullmatch(factor):
            problems.append(f"{label}: downstream_factor_digest is invalid")
        else:
            factor_key = (frame_id, seed, initializer, replicate)
            previous = factor_digests.setdefault(factor_key, factor)
            if factor != previous:
                problems.append(
                    f"{label}: downstream config changes across field families for {factor_key}"
                )
        if status != "ok":
            continue
        stage1 = row.get("stage1")
        downstream = row.get("downstream")
        if not isinstance(stage1, Mapping) or set(stage1) != set(predictor_names):
            problems.append(f"{label}: stage1 metrics do not match frozen predictors")
            continue
        if not isinstance(downstream, Mapping) or set(downstream) != set(response_names):
            problems.append(f"{label}: downstream metrics do not match frozen responses")
            continue
        try:
            current_stage1 = {
                name: _finite_number(stage1[name], f"{label}.stage1.{name}")
                for name in predictor_names
            }
            for name in response_names:
                _finite_number(downstream[name], f"{label}.downstream.{name}")
        except ProtocolError as exc:
            problems.append(str(exc))
            continue
        unit_key = (frame_id, family_id)
        previous_stage1 = stage1_values.setdefault(unit_key, current_stage1)
        if current_stage1 != previous_stage1:
            problems.append(f"{label}: Stage-1 metrics drift across downstream replicates")
    return problems


def aa_replay_result(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    aa = protocol["aa_replay"]
    base_key = (
        aa["frame_id"],
        aa["family_id"],
        aa["seed"],
        aa["initializer"],
        aa["primary_replicate"],
    )
    replay_key = (*base_key[:4], aa["replay_replicate"])
    by_key = _index_rows(rows)
    base = by_key.get(base_key)
    replay = by_key.get(replay_key)
    checks: dict[str, bool] = {
        "primary_present": base is not None,
        "replay_present": replay is not None,
    }
    deltas: dict[str, float | None] = {}
    if base is None or replay is None:
        return {"passed": False, "checks": checks, "metric_abs_deltas": deltas}
    checks["both_ok"] = base.get("status") == replay.get("status") == "ok"
    checks["field_manifest_exact"] = (
        base.get("field_manifest_sha256") == replay.get("field_manifest_sha256")
    )
    checks["field_semantics_exact"] = (
        base.get("field_semantic_digest") == replay.get("field_semantic_digest")
    )
    checks["downstream_factor_exact"] = (
        base.get("downstream_factor_digest") == replay.get("downstream_factor_digest")
    )
    tolerances = aa["metric_abs_tolerance"]
    for kind in ("stage1", "downstream"):
        left = base.get(kind, {})
        right = replay.get(kind, {})
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            checks[f"{kind}_metrics_present"] = False
            continue
        checks[f"{kind}_metrics_present"] = True
        for name in sorted(set(left) | set(right)):
            if name not in tolerances:
                continue
            try:
                delta = abs(float(left[name]) - float(right[name]))
            except (KeyError, TypeError, ValueError):
                delta = None
            deltas[name] = delta
            tolerance = tolerances.get(name)
            checks[f"metric_{name}_within_tolerance"] = (
                delta is not None
                and isinstance(tolerance, (int, float))
                and not isinstance(tolerance, bool)
                and delta <= float(tolerance)
            )
    return {
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "metric_abs_deltas": deltas,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = average
        position = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    x_ranks = np.asarray(_average_ranks([float(value) for value in x]), dtype=np.float64)
    y_ranks = np.asarray(_average_ranks([float(value) for value in y]), dtype=np.float64)
    if float(x_ranks.std()) == 0.0 or float(y_ranks.std()) == 0.0:
        return None
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def _direction(spec: Mapping[str, Any]) -> float:
    return 1.0 if spec["direction"] == "higher" else -1.0


def _aggregate_units(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    primary_replicate = protocol["aa_replay"]["primary_replicate"]
    predictor_specs = {spec["name"]: spec for spec in protocol["predictors"]}
    response_specs = {spec["name"]: spec for spec in protocol["responses"]}
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    expected = set(expected_cell_keys(protocol))
    for row in rows:
        if row.get("status") != "ok" or row.get("replicate_id") != primary_replicate:
            continue
        stage1 = row.get("stage1")
        downstream = row.get("downstream")
        if (
            not isinstance(stage1, Mapping)
            or set(stage1) != set(predictor_specs)
            or not isinstance(downstream, Mapping)
            or set(downstream) != set(response_specs)
        ):
            continue
        try:
            for value in (*stage1.values(), *downstream.values()):
                _finite_number(value, "result metric")
        except ProtocolError:
            continue
        try:
            cell_key = _row_key(row)
            key = (str(row["capture_id"]), str(row["frame_id"]), str(row["family_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        if cell_key not in expected:
            continue
        grouped.setdefault(key, []).append(row)
    units = []
    for (capture_id, frame_id, family_id), group in sorted(grouped.items()):
        first = group[0]
        stage1 = {
            name: float(first["stage1"][name])
            for name in predictor_specs
        }
        downstream = {
            name: float(np.mean([float(row["downstream"][name]) for row in group]))
            for name in response_specs
        }
        units.append(
            {
                "capture_id": capture_id,
                "frame_id": frame_id,
                "family_id": family_id,
                "cells": len(group),
                "stage1": stage1,
                "stage1_utility": {
                    name: _direction(predictor_specs[name]) * value
                    for name, value in stage1.items()
                },
                "downstream": downstream,
                "downstream_utility": {
                    name: _direction(response_specs[name]) * value
                    for name, value in downstream.items()
                },
            }
        )
    return units


def _ranking_rows(protocol: Mapping[str, Any], units: Sequence[Mapping[str, Any]]) -> list[dict]:
    primary = next(spec["name"] for spec in protocol["responses"] if spec["primary"])
    by_frame: dict[str, list[Mapping[str, Any]]] = {}
    for unit in units:
        by_frame.setdefault(str(unit["frame_id"]), []).append(unit)
    rows = []
    for frame_id, frame_units in sorted(by_frame.items()):
        response_values = [float(unit["downstream_utility"][primary]) for unit in frame_units]
        response_ranks = _average_ranks(response_values)
        for predictor in protocol["predictors"]:
            name = predictor["name"]
            predictor_values = [float(unit["stage1_utility"][name]) for unit in frame_units]
            predictor_ranks = _average_ranks(predictor_values)
            for unit, predictor_rank, response_rank in zip(
                frame_units, predictor_ranks, response_ranks
            ):
                rows.append(
                    {
                        "capture_id": unit["capture_id"],
                        "frame_id": frame_id,
                        "family_id": unit["family_id"],
                        "predictor": name,
                        "predictor_value": unit["stage1"][name],
                        "predictor_utility": unit["stage1_utility"][name],
                        "predictor_rank": predictor_rank,
                        "response": primary,
                        "response_value": unit["downstream"][primary],
                        "response_utility": unit["downstream_utility"][primary],
                        "response_rank": response_rank,
                        "downstream_cells": unit["cells"],
                    }
                )
    return rows


def _rho(rows: Sequence[Mapping[str, Any]]) -> float | None:
    return spearman(
        [float(row["predictor_rank"]) for row in rows],
        [float(row["response_rank"]) for row in rows],
    )


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    by_capture: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_capture.setdefault(str(row["capture_id"]), []).append(row)
    capture_ids = sorted(by_capture)
    if not capture_ids:
        return {"available": False, "replicates": 0, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        selected = rng.integers(0, len(capture_ids), size=len(capture_ids))
        sample = [
            row
            for index in selected
            for row in by_capture[capture_ids[int(index)]]
        ]
        value = _rho(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return {"available": False, "replicates": 0, "low": None, "high": None}
    low, high = np.quantile(np.asarray(values), [0.025, 0.975])
    return {
        "available": True,
        "replicates": len(values),
        "low": float(low),
        "high": float(high),
    }


def _lofo_rows(ranking_rows: Sequence[Mapping[str, Any]], predictor: str) -> list[dict]:
    selected = [row for row in ranking_rows if row["predictor"] == predictor]
    frames = sorted({str(row["frame_id"]) for row in selected})
    output = []
    for heldout in frames:
        train = [row for row in selected if row["frame_id"] != heldout]
        test = [row for row in selected if row["frame_id"] == heldout]
        predictor_best = {
            row["family_id"]
            for row in test
            if row["predictor_rank"] == min(item["predictor_rank"] for item in test)
        }
        response_best = {
            row["family_id"]
            for row in test
            if row["response_rank"] == min(item["response_rank"] for item in test)
        }
        output.append(
            {
                "predictor": predictor,
                "heldout_frame": heldout,
                "train_frames": len(frames) - 1,
                "train_spearman": _rho(train),
                "heldout_spearman": _rho(test),
                "heldout_top1_overlap": bool(predictor_best & response_best),
                "predictor_top1": sorted(predictor_best),
                "response_top1": sorted(response_best),
            }
        )
    return output


def analyze(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compute the frozen paired analysis; no view is ever an independent replicate."""
    row_problems = validate_result_rows(protocol, rows)
    expected = set(expected_cell_keys(protocol))
    by_key = dict(_index_rows(rows))
    missing = sorted(expected - set(by_key))
    error_keys = sorted(key for key, row in by_key.items() if row.get("status") == "error")
    aa = aa_replay_result(protocol, rows)
    units = _aggregate_units(protocol, rows)
    rankings = _ranking_rows(protocol, units)
    correlations = []
    lofo = []
    for predictor_index, predictor_spec in enumerate(protocol["predictors"]):
        predictor = predictor_spec["name"]
        selected = [row for row in rankings if row["predictor"] == predictor]
        per_frame = []
        for frame_id in sorted({row["frame_id"] for row in selected}):
            frame_rows = [row for row in selected if row["frame_id"] == frame_id]
            per_frame.append({"frame_id": frame_id, "spearman": _rho(frame_rows)})
        bootstrap = _cluster_bootstrap(
            selected,
            replicates=int(protocol["analysis"]["bootstrap_replicates"]),
            seed=int(protocol["analysis"]["bootstrap_seed"]) + predictor_index,
        )
        predictor_lofo = _lofo_rows(rankings, predictor)
        lofo.extend(predictor_lofo)
        lofo_agreement = (
            float(np.mean([row["heldout_top1_overlap"] for row in predictor_lofo]))
            if predictor_lofo
            else None
        )
        correlations.append(
            {
                "predictor": predictor,
                "pooled_within_frame_spearman": _rho(selected),
                "frames": len({row["frame_id"] for row in selected}),
                "captures": len({row["capture_id"] for row in selected}),
                "units": len(selected),
                "per_frame": per_frame,
                "cluster_bootstrap": bootstrap,
                "lofo_top1_agreement": lofo_agreement,
            }
        )

    capture_count = len({unit["capture_id"] for unit in units})
    frame_count = len({unit["frame_id"] for unit in units})
    family_count = len({unit["family_id"] for unit in units})
    thresholds = protocol["analysis"]
    scope_ok = (
        capture_count >= int(thresholds["minimum_capture_groups"])
        and frame_count >= int(thresholds["minimum_frames"])
        and family_count >= int(thresholds["minimum_family_count"])
    )
    complete = not missing and not error_keys and not row_problems
    gates = {}
    for correlation in correlations:
        rho = correlation["pooled_within_frame_spearman"]
        lower = correlation["cluster_bootstrap"]["low"]
        agreement = correlation["lofo_top1_agreement"]
        gates[correlation["predictor"]] = {
            "spearman": rho,
            "spearman_pass": rho is not None and rho >= thresholds["minimum_spearman"],
            "bootstrap_lower": lower,
            "bootstrap_pass": (
                lower is not None and lower >= thresholds["minimum_bootstrap_lower"]
            ),
            "lofo_top1_agreement": agreement,
            "lofo_pass": (
                agreement is not None
                and agreement >= thresholds["minimum_lofo_top1_agreement"]
            ),
        }
        gates[correlation["predictor"]]["passed"] = all(
            gates[correlation["predictor"]][name]
            for name in ("spearman_pass", "bootstrap_pass", "lofo_pass")
        )

    decision_state: str
    selected_predictor: str | None = None
    reason: str
    if not aa["passed"]:
        decision_state = "question_unavailable"
        reason = "A/A replay did not prove adapter/config/semantic identity"
    elif not complete:
        decision_state = "question_unavailable"
        reason = "frozen expected cells are missing, erroneous, or binding-invalid"
    elif not scope_ok:
        decision_state = "question_unavailable"
        reason = "capture/frame/family scope is below the preregistered minimum"
    else:
        selected_predictor = next(
            (
                predictor
                for predictor in thresholds["selection_priority"]
                if gates[predictor]["passed"]
            ),
            None,
        )
        if selected_predictor is None:
            decision_state = "require_downstream_evaluation"
            reason = "no preregistered Stage-1 predictor passed all validity gates"
        else:
            decision_state = "select_stage1_surrogate"
            reason = "the first preregistered passing predictor was selected without blending"
    primary_response = next(
        spec["name"] for spec in protocol["responses"] if spec["primary"]
    )
    return {
        "schema": "structsplat.bench019.analysis.v1",
        "protocol_sha256": protocol.get("protocol_sha256"),
        "primary_response": primary_response,
        "expected_cells": len(expected),
        "observed_cells": len(by_key),
        "missing_cells": [list(key) for key in missing],
        "error_cells": [list(key) for key in error_keys],
        "row_validation_problems": row_problems,
        "aa_replay": aa,
        "units": units,
        "rankings": rankings,
        "correlations": correlations,
        "leave_one_frame_out": lofo,
        "scope": {
            "claim_scope": protocol["claim_scope"],
            "captures": capture_count,
            "frames": frame_count,
            "families": family_count,
            "scope_gate_passed": scope_ok,
        },
        "decision": {
            "state": decision_state,
            "selected_predictor": selected_predictor,
            "primary_response": primary_response,
            "reason": reason,
            "gates": gates,
            "no_posthoc_metric_blend": True,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not fields:
            stream.write("")
            return
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _materialize_artifact(
    record: Mapping[str, Any], destination: Path, *, label: str
) -> dict[str, Any]:
    source = _validate_artifact(record, Path.cwd(), label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": destination.as_posix(),
        "sha256": sha256_file(destination),
        "bytes": int(destination.stat().st_size),
    }


def _portable_rows(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    outdir: Path,
) -> list[dict[str, Any]]:
    expected = expected_cell_keys(protocol)
    by_key = {key: dict(row) for key, row in _index_rows(rows).items()}
    portable = []
    for key in expected:
        row = copy.deepcopy(by_key.get(key))
        if row is None:
            frame_id, family_id, seed, initializer, replicate = key
            capture_id = _frame_index(protocol)[frame_id][0]
            row = {
                "schema": ROW_SCHEMA,
                "status": "missing",
                "error": "expected cell absent from result export",
                "capture_id": capture_id,
                "frame_id": frame_id,
                "family_id": family_id,
                "seed": seed,
                "initializer": initializer,
                "replicate_id": replicate,
                "stage1": {},
                "downstream": {},
                "artifacts": {},
            }
        if row.get("status") == "ok":
            artifacts = row.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise ProtocolError(f"cell {key} has no artifact map")
            copied = {}
            cell_dir = outdir / "artifacts" / _slug("__".join(map(str, key)))
            for name in REQUIRED_CELL_ARTIFACTS:
                source_record = _mapping(artifacts.get(name), f"cell {key} artifact {name}")
                source_path = _artifact_path(source_record, Path.cwd())
                suffix = source_path.suffix or ".bin"
                destination = cell_dir / f"{name}{suffix}"
                descriptor = _materialize_artifact(
                    source_record, destination, label=f"cell {key} artifact {name}"
                )
                descriptor["path"] = destination.relative_to(outdir).as_posix()
                copied[name] = descriptor
            row["artifacts"] = copied
        portable.append(row)
    return portable


def _copy_protocol_bindings(
    protocol: Mapping[str, Any], outdir: Path
) -> list[dict[str, Any]]:
    bindings = []
    for ordinal, (label, record) in enumerate(_artifact_records(protocol)):
        source = _validate_artifact(record, Path.cwd(), label)
        suffix = source.suffix or ".bin"
        destination = outdir / "bindings" / f"{ordinal:03d}_{_slug(label)}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        bindings.append(
            {
                "label": label,
                "path": destination.relative_to(outdir).as_posix(),
                "sha256": sha256_file(destination),
                "bytes": int(destination.stat().st_size),
            }
        )
    return bindings


def write_report(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    outdir: str | Path,
    *,
    command: str,
    allow_review_protocol: bool = False,
) -> dict[str, Any]:
    """Write an immutable-style portable report; refuse to merge into a non-empty directory."""
    validate_protocol(protocol, require_frozen=not allow_review_protocol)
    out = Path(outdir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ProtocolError(f"report output is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    analysis = analyze(protocol, rows)
    portable_rows = _portable_rows(protocol, rows, out)
    bindings = _copy_protocol_bindings(protocol, out)
    _write_json(out / "protocol.json", protocol)
    _write_json(out / "analysis.json", analysis)
    _write_json(out / "decision.json", analysis["decision"])
    _write_json(out / "metrics.json", portable_rows)
    (out / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in portable_rows),
        encoding="utf-8",
    )
    _write_csv(out / "metrics.csv", portable_rows)
    _write_csv(out / "rankings.csv", analysis["rankings"])
    _write_csv(out / "correlations.csv", analysis["correlations"])
    _write_csv(out / "leave_one_frame_out.csv", analysis["leave_one_frame_out"])
    missing_rows = [
        {
            "frame_id": key[0],
            "family_id": key[1],
            "seed": key[2],
            "initializer": key[3],
            "replicate_id": key[4],
        }
        for key in analysis["missing_cells"]
    ]
    _write_csv(out / "missing_cells.csv", missing_rows)

    manifest = {
        "schema": REPORT_SCHEMA,
        "command": command,
        "protocol_file": "protocol.json",
        "protocol_file_sha256": sha256_file(out / "protocol.json"),
        "protocol_sha256": protocol.get("protocol_sha256"),
        "design_sha256": protocol["design_sha256"],
        "repositories": [
            {
                "name": repository["name"],
                "commit": repository["commit"],
                "branch": repository["branch"],
                "dirty": repository["dirty"],
                "status_sha256": repository["status_sha256"],
                "environment_sha256": repository["environment"]["sha256"],
            }
            for repository in protocol["repositories"]
        ],
        "variants": sorted(
            {family["id"] for capture in protocol["captures"] for frame in capture["frames"] for family in frame["families"]}
        ),
        "seeds": list(protocol["downstream"]["seeds"]),
        "frames": sorted(_frame_index(protocol)),
        "bindings": bindings,
        "analysis_files": [
            "analysis.json",
            "decision.json",
            "rankings.csv",
            "correlations.csv",
            "leave_one_frame_out.csv",
            "missing_cells.csv",
        ],
        "decision": analysis["decision"],
        "claim_ready": (
            protocol["state"] == "frozen"
            and not analysis["missing_cells"]
            and not analysis["error_cells"]
            and not analysis["row_validation_problems"]
            and analysis["aa_replay"]["passed"]
        ),
    }
    _write_json(out / "manifest.json", manifest)

    links = [
        "manifest.json",
        "protocol.json",
        "metrics.json",
        "metrics.jsonl",
        "metrics.csv",
        *manifest["analysis_files"],
        *[binding["path"] for binding in bindings],
    ]
    for row in portable_rows:
        for descriptor in row.get("artifacts", {}).values():
            if isinstance(descriptor, Mapping) and isinstance(descriptor.get("path"), str):
                links.append(descriptor["path"])
    correlation_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['predictor']))}</td>"
        f"<td>{html.escape(str(row['pooled_within_frame_spearman']))}</td>"
        f"<td>{html.escape(str(row['cluster_bootstrap']['low']))}</td>"
        f"<td>{html.escape(str(row['lofo_top1_agreement']))}</td>"
        "</tr>"
        for row in analysis["correlations"]
    )
    link_html = "\n".join(
        f'<li><a href="{html.escape(link)}">{html.escape(link)}</a></li>'
        for link in dict.fromkeys(links)
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>BENCH-019 Stage-1 downstream objective</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.35rem}}</style></head>
<body><h1>BENCH-019 Stage-1 downstream objective</h1>
<p>Decision: <strong>{html.escape(analysis['decision']['state'])}</strong> —
{html.escape(analysis['decision']['reason'])}</p>
<p>Expected/observed cells: {analysis['expected_cells']} / {analysis['observed_cells']};
captures: {analysis['scope']['captures']}; frames: {analysis['scope']['frames']};
families: {analysis['scope']['families']}; A/A pass: {analysis['aa_replay']['passed']}.</p>
<table><thead><tr><th>Predictor</th><th>Spearman</th><th>cluster CI low</th><th>LOFO top-1</th></tr></thead>
<tbody>{correlation_rows}</tbody></table>
<h2>Portable files and artifacts</h2><ul>{link_html}</ul></body></html>
"""
    (out / "index.html").write_text(document, encoding="utf-8")
    return manifest


def protocol_template() -> dict[str, Any]:
    """Return an intentionally non-runnable authoring template (paths must be replaced)."""
    artifact = {"path": "REPLACE_WITH_FILE"}
    semantics = {
        "provider": "native",
        "equation": "additive_sum",
        "blend_mode": "additive",
        "alpha_policy": "packed_alpha",
        "coordinate_convention": "top-left pixel center is (0.5,0.5)",
        "semantic_digest": "0" * 64,
    }
    return {
        "schema": PROTOCOL_SCHEMA,
        "task_id": "BENCH-019",
        "state": "draft",
        "driver": "replace-driver",
        "claim_scope": "general",
        "repositories": [
            {"name": "structsplat", "root": ".", "environment": artifact},
            {"name": "realtime-gs", "root": "../realtime-gs", "environment": artifact},
        ],
        "captures": [
            {
                "id": "capture_a",
                "frames": [
                    {
                        "id": "frame_a",
                        "pixels": artifact,
                        "masks": artifact,
                        "cameras": artifact,
                        "split": {"train": ["view_train"], "heldout": ["view_heldout"]},
                        "families": [
                            {
                                "id": "family_a",
                                "field_manifest": artifact,
                                "stage1_metrics": artifact,
                                "semantics": semantics,
                            }
                        ],
                    }
                ],
            }
        ],
        "downstream": {
            "task_manifest": artifact,
            "dataset_manifest": artifact,
            "environment": artifact,
            "schedule_config": artifact,
            "command": ["python", "REPLACE_WITH_REALTIME_DRIVER", "run"],
            "working_directory": "/REPLACE_WITH_REALTIME_REPOSITORY",
            "outcome_root": "/REPLACE_WITH_EMPTY_OUTCOME_ROOT",
            "seeds": [19001, 19002, 19003],
            "initializers": ["fixed_initializer"],
            "result_schema": RESULT_SCHEMA,
        },
        "predictors": [
            {"name": "foreground_psnr", "direction": "higher"},
            {"name": "cold_query_error", "direction": "lower"},
        ],
        "responses": [
            {"name": "heldout_lpips", "direction": "lower", "primary": True},
            {"name": "fit_seconds", "direction": "lower", "primary": False},
        ],
        "analysis": {
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 19019,
            "minimum_capture_groups": 3,
            "minimum_frames": 2,
            "minimum_family_count": 3,
            "minimum_spearman": 0.8,
            "minimum_bootstrap_lower": 0.0,
            "minimum_lofo_top1_agreement": 0.67,
            "selection_priority": ["foreground_psnr", "cold_query_error"],
            "missing_policy": "fail_closed",
        },
        "aa_replay": {
            "frame_id": "frame_a",
            "family_id": "family_a",
            "seed": 19001,
            "initializer": "fixed_initializer",
            "primary_replicate": "primary",
            "replay_replicate": "aa",
            "metric_abs_tolerance": {
                "foreground_psnr": 0.0,
                "cold_query_error": 1e-7,
                "heldout_lpips": 1e-6,
            },
        },
    }


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"{path} must contain a JSON object")
    return value


def _write_stdout(value: object) -> None:
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
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--protocol", type=Path, required=True)
    analyze_parser.add_argument("--rows", type=Path, required=True)
    analyze_parser.add_argument("--outdir", type=Path, required=True)
    analyze_parser.add_argument("--diagnostic-review-protocol", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "template":
        _write_json(args.output, protocol_template())
        return 0
    if args.command == "prepare-review":
        protocol = prepare_review(_load_object(args.draft), base=args.draft.parent)
        _write_json(args.output, protocol)
        return 0
    if args.command == "review-template":
        protocol = _load_object(args.protocol)
        _write_json(args.output, review_template(protocol))
        return 0
    if args.command == "finalize":
        protocol = finalize_protocol(
            _load_object(args.reviewed), args.review, base=args.reviewed.parent
        )
        _write_json(args.output, protocol)
        return 0
    if args.command == "validate":
        protocol = _load_object(args.protocol)
        validate_protocol(
            protocol,
            base=args.protocol.parent,
            require_frozen=args.require_frozen,
        )
        print(f"BENCH-019 protocol OK: {protocol.get('protocol_sha256', protocol['design_sha256'])}")
        return 0
    if args.command == "plan":
        protocol = _load_object(args.protocol)
        validate_protocol(protocol, base=args.protocol.parent, require_frozen=True)
        _write_stdout(
            {
                "protocol_sha256": protocol["protocol_sha256"],
                "canonical_downstream_command": protocol["downstream"]["command"],
                "outcome_root": protocol["downstream"]["outcome_root"],
                "cells": [list(key) for key in expected_cell_keys(protocol)],
            }
        )
        return 0
    if args.command == "analyze":
        protocol = _load_object(args.protocol)
        rows = load_rows(args.rows)
        command = " ".join(os.sys.argv)
        manifest = write_report(
            protocol,
            rows,
            args.outdir,
            command=command,
            allow_review_protocol=args.diagnostic_review_protocol,
        )
        _write_stdout(
            {
                "report": str(args.outdir.resolve()),
                "decision": manifest["decision"],
                "claim_ready": manifest["claim_ready"],
            }
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
