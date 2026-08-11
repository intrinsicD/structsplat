#!/usr/bin/env python3
"""Validate a portable report produced by StructSplat's maintained workflows.

This is the evidence-handoff gate for ``scripts/convert.py``, ``benchmark.py``,
``ablation.py``, ``stage_search.py``, the BENCH-019/BENCH-020 experiment controllers, and the
task-scoped HIER contraction/repair/elimination diagnostics, and CORE-019's multi-view coherent-
depth diagnostic. It checks
the applicable manifest/metrics contract, clean source identity, per-cell artifacts, finite
metrics, cross-format agreement, and every local HTML link.

Dirty-source or error-cell reports remain useful diagnostics, but they are not results-bearing
by default. ``--allow-dirty`` and ``--allow-error-cells`` make those limitations explicit.

Run: python scripts/check_report_bundle.py RESULTS_DIR
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import unquote, urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = (
    "manifest.json",
    "metrics.json",
    "metrics.jsonl",
    "metrics.csv",
    "index.html",
)
REQUIRED_OK_ARTIFACTS = (
    "target_png",
    "reconstruction_png",
    "error_png",
    "field_npz",
    "history_json",
    "config_json",
)
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
CLEAN_STATUS_SHA256 = hashlib.sha256(b"").hexdigest()
BENCH019_REPORT_SCHEMA = "structsplat.bench019.report.v1"
BENCH019_PROTOCOL_SCHEMA = "structsplat.bench019.protocol.v1"
BENCH019_ROW_SCHEMA = "structsplat.bench019.cell.v1"
BENCH019_REQUIRED_ARTIFACTS = (
    "field",
    "history",
    "config",
    "target",
    "reconstruction",
    "error",
)
BENCH020_REPORT_SCHEMA = "structsplat.bench020.report.v1"
HIER005_CONTRACTION_SCHEMA = "structsplat.hier005_pixel_contraction.diagnostic.v1"
HIER005_REPAIR_SCHEMA = "structsplat.hier005_artifact_repair.diagnostic.v1"
HIER008_OVERLAP_SCHEMA = "structsplat.hier008_overlap_elimination.diagnostic.v1"
HIER009_DYNAMIC_SCHEMA = "structsplat.hier009_dynamic_overlap_recovery.diagnostic.v1"
HIER010_REFINEMENT_SCHEMA = "structsplat.hier010_residual_anchor_projection.diagnostic.v1"
HIER011_EXCHANGE_SCHEMA = "structsplat.hier011_guarded_residual_column_exchange.diagnostic.v1"
HIER012_PROJECTION_SCHEMA = "structsplat.hier012_global_appearance_projection.diagnostic.v1"
HIER013_DEVELOPMENT_SCHEMA = "structsplat.hier013_global_projection_development.diagnostic.v1"
HIER015_GEOMETRY_SCHEMA = "structsplat.hier015_geometry_escape.diagnostic.v1"
HIER016_TAIL_SCHEMA = "structsplat.hier016_normalized_tail_repair.diagnostic.v1"
HIER017_EPSILON_SCHEMA = "structsplat.hier017_normalization_epsilon.diagnostic.v1"
HIER018_BACKGROUND_SCHEMA = "structsplat.hier018_counted_background.diagnostic.v1"
HIER019_TAIL_SCHEMA = "structsplat.hier019_confidence_tail.diagnostic.v1"
HIER020_SPARSE_TAIL_SCHEMA = "structsplat.hier020_sparse_pixel_safe_tail.diagnostic.v1"
HIER021_SOURCE_PATCH_SCHEMA = "structsplat.hier021_source_patch_tail.diagnostic.v1"
CORE019_REPORT_SCHEMA = "core019.coherent_depth.manifest.v1"
HIER005_REPORT_SCHEMAS = frozenset(
    {
        HIER005_CONTRACTION_SCHEMA,
        HIER005_REPAIR_SCHEMA,
        HIER008_OVERLAP_SCHEMA,
        HIER009_DYNAMIC_SCHEMA,
        HIER010_REFINEMENT_SCHEMA,
        HIER011_EXCHANGE_SCHEMA,
        HIER012_PROJECTION_SCHEMA,
        HIER013_DEVELOPMENT_SCHEMA,
    }
)
HIER015_PLUS_REPORT_SCHEMAS = frozenset(
    {
        HIER015_GEOMETRY_SCHEMA,
        HIER016_TAIL_SCHEMA,
        HIER017_EPSILON_SCHEMA,
        HIER018_BACKGROUND_SCHEMA,
        HIER019_TAIL_SCHEMA,
        HIER020_SPARSE_TAIL_SCHEMA,
        HIER021_SOURCE_PATCH_SCHEMA,
    }
)


class _LocalLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        keys = {"href", "src"}
        for name, value in attrs:
            if name in keys and value:
                self.links.append(value)


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_invalid_constant,
    )


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except (FileNotFoundError, ValueError):
        return False


def _artifact_path(root: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    value = Path(raw)
    if value.is_absolute():
        return None
    candidates = [root / value, REPOSITORY_ROOT / value]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0 and _contained(candidate, root):
            return candidate.resolve()
    return None


def _check_repository_identity(
    repository: Any,
    source: str,
    problems: list[str],
    *,
    allow_dirty: bool,
) -> None:
    if not isinstance(repository, dict):
        problems.append(f"{source}: repository identity must be an object")
        return
    commit = repository.get("commit")
    branch = repository.get("branch")
    status_hash = repository.get("status_sha256")
    dirty = repository.get("dirty")
    if not isinstance(commit, str) or not HEX_40.fullmatch(commit):
        problems.append(f"{source}: repository.commit must be a 40-character Git SHA")
    if not isinstance(branch, str) or branch == "unavailable":
        problems.append(f"{source}: repository.branch is missing or unavailable")
    if not isinstance(status_hash, str) or not HEX_64.fullmatch(status_hash):
        problems.append(f"{source}: repository.status_sha256 must be a SHA-256")
    if not isinstance(dirty, bool):
        problems.append(f"{source}: repository.dirty must be a boolean")
    else:
        if dirty and status_hash == CLEAN_STATUS_SHA256:
            problems.append(f"{source}: dirty repository has the clean status hash")
        if not dirty and isinstance(status_hash, str) and status_hash != CLEAN_STATUS_SHA256:
            problems.append(f"{source}: clean repository has a non-clean status hash")
        if dirty and not allow_dirty:
            problems.append(
                f"{source}: repository was dirty; rerun from a clean commit or use "
                "--allow-dirty for a non-claim diagnostic"
            )


def _load_jsonl(path: Path, problems: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=_invalid_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            problems.append(f"metrics.jsonl:{lineno}: invalid JSON: {exc}")
            continue
        if not isinstance(value, dict):
            problems.append(f"metrics.jsonl:{lineno}: row must be an object")
            continue
        rows.append(value)
    return rows


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    if value is None:
        return ""
    return str(value)


def _check_html(root: Path, problems: list[str]) -> set[Path]:
    index = root / "index.html"
    try:
        text = index.read_text(encoding="utf-8")
    except OSError as exc:
        problems.append(f"index.html cannot be read: {exc}")
        return set()
    parser = _LocalLinkParser()
    try:
        parser.feed(text)
    except Exception as exc:  # pragma: no cover - defensive HTMLParser boundary
        problems.append(f"index.html cannot be parsed: {exc}")
        return set()

    linked_paths: set[Path] = set()
    for raw in parser.links:
        split = urlsplit(raw)
        if split.scheme or split.netloc or raw.startswith("/"):
            problems.append(f"index.html has non-portable absolute/external link: {raw}")
            continue
        if not split.path:
            continue
        target = root / unquote(split.path)
        if not _contained(target, root):
            problems.append(f"index.html link escapes or is missing from the bundle: {raw}")
            continue
        linked_paths.add(target.resolve())

    for name in ("manifest.json", "metrics.json", "metrics.jsonl", "metrics.csv"):
        if (root / name).resolve() not in linked_paths:
            problems.append(f"index.html does not link required bundle file {name}")
    if not re.search(r"\d", text):
        problems.append("index.html contains no numeric result summary")
    return linked_paths


def _check_json_csv_projection(
    root: Path,
    metrics: list[dict[str, Any]],
    problems: list[str],
) -> None:
    """Check the common JSON/JSONL/CSV projection shared by both report schemas."""
    jsonl_rows = _load_jsonl(root / "metrics.jsonl", problems)
    if jsonl_rows != metrics:
        problems.append("metrics.jsonl rows do not exactly match metrics.json")
    try:
        with (root / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            csv_fields = reader.fieldnames or []
            csv_rows = list(reader)
    except (OSError, csv.Error) as exc:
        problems.append(f"metrics.csv is invalid: {exc}")
        return
    expected_fields = sorted(
        {key for row in metrics for key in row if key not in {"curves", "snapshots"}}
    )
    if csv_fields != expected_fields:
        problems.append("metrics.csv columns do not match the canonical metrics.json projection")
        return
    if len(csv_rows) != len(metrics):
        problems.append(
            f"metrics.csv row count {len(csv_rows)} does not match metrics.json "
            f"row count {len(metrics)}"
        )
        return
    for index, (csv_row, json_row) in enumerate(zip(csv_rows, metrics)):
        for field in expected_fields:
            expected = _csv_value(json_row.get(field))
            if csv_row.get(field) != expected:
                problems.append(
                    f"metrics.csv row {index} field {field!r} differs from metrics.json"
                )


def _relative_descriptor_path(
    root: Path,
    descriptor: Any,
    label: str,
    problems: list[str],
) -> Path | None:
    if not isinstance(descriptor, dict):
        problems.append(f"{label}: artifact descriptor must be an object")
        return None
    if set(descriptor) != {"path", "sha256", "bytes"}:
        problems.append(f"{label}: artifact descriptor must contain path/sha256/bytes")
        return None
    raw = descriptor.get("path")
    digest = descriptor.get("sha256")
    size = descriptor.get("bytes")
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        problems.append(f"{label}: artifact path must be non-empty and relative")
        return None
    if not isinstance(digest, str) or not HEX_64.fullmatch(digest):
        problems.append(f"{label}: artifact SHA-256 is missing or malformed")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        problems.append(f"{label}: artifact byte count must be a non-negative integer")
    path = root / raw
    if not path.is_file() or not _contained(path, root):
        problems.append(f"{label}: artifact is missing or escapes the report bundle")
        return None
    if isinstance(size, int) and path.stat().st_size != size:
        problems.append(f"{label}: artifact byte count differs")
    if isinstance(digest, str) and HEX_64.fullmatch(digest) and _sha256(path) != digest:
        problems.append(f"{label}: artifact SHA-256 differs")
    return path.resolve()


def _canonical_digest_without(payload: dict[str, Any], key: str) -> str:
    value = dict(payload)
    value.pop(key, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bench019_protocol_bindings(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Project the protocol's bound files using the writer's stable binding labels."""
    bindings: dict[str, dict[str, Any]] = {}
    repositories = protocol.get("repositories")
    if isinstance(repositories, list):
        for index, repository in enumerate(repositories):
            if not isinstance(repository, dict):
                continue
            label = f"repository_{repository.get('name', index)}_environment"
            if isinstance(repository.get("environment"), dict):
                bindings[label] = repository["environment"]
    captures = protocol.get("captures")
    if isinstance(captures, list):
        for capture_index, capture in enumerate(captures):
            if not isinstance(capture, dict) or not isinstance(capture.get("frames"), list):
                continue
            for frame_index, frame in enumerate(capture["frames"]):
                if not isinstance(frame, dict):
                    continue
                prefix = f"{capture.get('id', capture_index)}_{frame.get('id', frame_index)}"
                for name in ("pixels", "masks", "cameras"):
                    if isinstance(frame.get(name), dict):
                        bindings[f"{prefix}_{name}"] = frame[name]
                families = frame.get("families")
                if not isinstance(families, list):
                    continue
                for family_index, family in enumerate(families):
                    if not isinstance(family, dict):
                        continue
                    family_prefix = f"{prefix}_{family.get('id', family_index)}"
                    for name in ("field_manifest", "stage1_metrics"):
                        if isinstance(family.get(name), dict):
                            bindings[f"{family_prefix}_{name}"] = family[name]
    downstream = protocol.get("downstream")
    if isinstance(downstream, dict):
        for name in ("task_manifest", "dataset_manifest", "environment", "schedule_config"):
            if isinstance(downstream.get(name), dict):
                bindings[f"downstream_{name}"] = downstream[name]
    review = protocol.get("review")
    if isinstance(review, dict) and isinstance(review.get("artifact"), dict):
        bindings["prospective_review"] = review["artifact"]
    return bindings


def _check_bench019_bundle(
    root: Path,
    manifest: dict[str, Any],
    problems: list[str],
    *,
    allow_dirty: bool,
    allow_error_cells: bool,
) -> None:
    protocol: dict[str, Any] | None = None
    expected_bindings: dict[str, dict[str, Any]] = {}
    command = manifest.get("command")
    if not isinstance(command, str) or not command.strip():
        problems.append("manifest.json has no executed command")
    repositories = manifest.get("repositories")
    if not isinstance(repositories, list) or len(repositories) < 2:
        problems.append("BENCH-019 manifest must bind both source repositories")
    else:
        names: set[str] = set()
        for index, repository in enumerate(repositories):
            _check_repository_identity(
                repository,
                f"manifest.json repositories[{index}]",
                problems,
                allow_dirty=allow_dirty,
            )
            if not isinstance(repository, dict) or not isinstance(repository.get("name"), str):
                problems.append(f"manifest.json repositories[{index}] has no name")
            elif repository["name"] in names:
                problems.append("manifest.json repository names must be unique")
            else:
                names.add(repository["name"])

    protocol_raw = manifest.get("protocol_file")
    if not isinstance(protocol_raw, str) or Path(protocol_raw).is_absolute():
        problems.append("manifest.json protocol_file must be a relative path")
        protocol_path = None
    else:
        candidate = root / protocol_raw
        protocol_path = candidate if candidate.is_file() and _contained(candidate, root) else None
        if protocol_path is None:
            problems.append("manifest.json protocol_file is missing or escapes the bundle")
    if protocol_path is not None:
        recorded_file_digest = manifest.get("protocol_file_sha256")
        if recorded_file_digest != _sha256(protocol_path):
            problems.append("manifest.json protocol_file_sha256 differs")
        try:
            loaded_protocol = _load_json(protocol_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            problems.append(f"protocol file is invalid: {exc}")
        else:
            if (
                not isinstance(loaded_protocol, dict)
                or loaded_protocol.get("schema") != BENCH019_PROTOCOL_SCHEMA
            ):
                problems.append("protocol file has the wrong BENCH-019 schema")
            else:
                protocol = loaded_protocol
                expected_bindings = _bench019_protocol_bindings(protocol)
            if protocol is not None and protocol.get("state") != "frozen" and not allow_dirty:
                problems.append("BENCH-019 protocol is not frozen")
            elif protocol is not None and protocol.get("state") == "frozen":
                recorded = protocol.get("protocol_sha256")
                if not isinstance(recorded, str) or recorded != _canonical_digest_without(
                    protocol, "protocol_sha256"
                ):
                    problems.append("protocol file self-digest differs")
                if manifest.get("protocol_sha256") != recorded:
                    problems.append("manifest.json binds a different protocol digest")

    bindings = manifest.get("bindings")
    required_paths: set[Path] = set()
    if not isinstance(bindings, list) or not bindings:
        problems.append("BENCH-019 manifest has no portable protocol bindings")
    else:
        labels: set[str] = set()
        observed_bindings: dict[str, dict[str, Any]] = {}
        for index, binding in enumerate(bindings):
            if not isinstance(binding, dict) or set(binding) != {
                "label",
                "path",
                "sha256",
                "bytes",
            }:
                problems.append(f"manifest.json bindings[{index}] has invalid fields")
                continue
            label = binding.get("label")
            if not isinstance(label, str) or not label or label in labels:
                problems.append(f"manifest.json bindings[{index}] has a missing/duplicate label")
            else:
                labels.add(label)
                observed_bindings[label] = binding
            descriptor = {key: binding[key] for key in ("path", "sha256", "bytes")}
            path = _relative_descriptor_path(
                root, descriptor, f"manifest.json bindings[{index}]", problems
            )
            if path is not None:
                required_paths.add(path)
        if expected_bindings:
            if set(observed_bindings) != set(expected_bindings):
                problems.append("portable binding labels differ from the frozen protocol")
            for label in sorted(set(observed_bindings) & set(expected_bindings)):
                expected = expected_bindings[label]
                observed = observed_bindings[label]
                if observed.get("sha256") != expected.get("sha256"):
                    problems.append(f"portable binding {label} has the wrong source SHA-256")
                if observed.get("bytes") != expected.get("bytes"):
                    problems.append(f"portable binding {label} has the wrong source byte count")

    try:
        metrics = _load_json(root / "metrics.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        problems.append(f"metrics.json is invalid: {exc}")
        return
    if (
        not isinstance(metrics, list)
        or not metrics
        or not all(isinstance(row, dict) for row in metrics)
    ):
        problems.append("metrics.json must contain a non-empty object-row list")
        return
    if not _finite(metrics):
        problems.append("metrics.json contains a non-finite numeric value")
    _check_json_csv_projection(root, metrics, problems)
    keys: set[tuple[Any, ...]] = set()
    for index, row in enumerate(metrics):
        label = f"metrics.json[{index}]"
        if row.get("schema") != BENCH019_ROW_SCHEMA:
            problems.append(f"{label}: wrong BENCH-019 row schema")
        status = row.get("status")
        if status not in {"ok", "error", "missing"}:
            problems.append(f"{label}: status must be ok, error, or missing")
        if status in {"error", "missing"} and not allow_error_cells:
            problems.append(
                f"{label}: {status} cell is not claim-ready; fix/rerun or use "
                "--allow-error-cells for diagnostics"
            )
        if status in {"error", "missing"} and not str(row.get("error", "")).strip():
            problems.append(f"{label}: non-ok cell must retain a diagnostic")
        key = tuple(
            repr(row.get(name))
            for name in ("frame_id", "family_id", "seed", "initializer", "replicate_id")
        )
        if key in keys:
            problems.append(f"{label}: duplicate stable cell key {key!r}")
        keys.add(key)
        if status != "ok":
            continue
        if not isinstance(row.get("stage1"), dict) or not row["stage1"]:
            problems.append(f"{label}: stage1 metrics must be a non-empty object")
        if not isinstance(row.get("downstream"), dict) or not row["downstream"]:
            problems.append(f"{label}: downstream metrics must be a non-empty object")
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(BENCH019_REQUIRED_ARTIFACTS):
            problems.append(f"{label}: artifacts do not match the BENCH-019 contract")
            continue
        for name in BENCH019_REQUIRED_ARTIFACTS:
            path = _relative_descriptor_path(
                root, artifacts[name], f"{label}.artifacts.{name}", problems
            )
            if path is not None:
                required_paths.add(path)

    analysis_files = manifest.get("analysis_files")
    if not isinstance(analysis_files, list) or not analysis_files:
        problems.append("manifest.json has no analysis_files")
    else:
        for index, raw in enumerate(analysis_files):
            if not isinstance(raw, str) or Path(raw).is_absolute():
                problems.append(f"manifest.json analysis_files[{index}] is not relative")
                continue
            path = root / raw
            if not path.is_file() or not _contained(path, root):
                problems.append(f"manifest.json analysis_files[{index}] is missing")
            else:
                required_paths.add(path.resolve())
    analysis: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    try:
        loaded_analysis = _load_json(root / "analysis.json")
        loaded_decision = _load_json(root / "decision.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        problems.append(f"BENCH-019 analysis/decision JSON is invalid: {exc}")
    else:
        if not isinstance(loaded_analysis, dict) or not isinstance(loaded_decision, dict):
            problems.append("BENCH-019 analysis and decision files must contain objects")
        else:
            analysis = loaded_analysis
            decision = loaded_decision
            if analysis.get("protocol_sha256") != manifest.get("protocol_sha256"):
                problems.append("analysis.json binds a different protocol digest")
            if analysis.get("decision") != decision or manifest.get("decision") != decision:
                problems.append("manifest/analysis/decision dispositions disagree")
            row_problems = analysis.get("row_validation_problems")
            missing = analysis.get("missing_cells")
            errors = analysis.get("error_cells")
            aa = analysis.get("aa_replay")
            if not allow_error_cells:
                if row_problems:
                    problems.append("analysis reports result-row binding problems")
                if missing:
                    problems.append("analysis reports missing expected cells")
                if errors:
                    problems.append("analysis reports error cells")
                if not isinstance(aa, dict) or aa.get("passed") is not True:
                    problems.append("analysis A/A replay did not pass")
            integrity_ready = bool(
                protocol is not None
                and protocol.get("state") == "frozen"
                and not row_problems
                and not missing
                and not errors
                and isinstance(aa, dict)
                and aa.get("passed") is True
            )
            if manifest.get("claim_ready") is not integrity_ready:
                problems.append("manifest.json claim_ready disagrees with analysis integrity")
    linked_paths = _check_html(root, problems)
    if protocol_path is not None:
        required_paths.add(protocol_path.resolve())
    for path in sorted(required_paths):
        if path not in linked_paths:
            problems.append(
                f"index.html does not expose BENCH-019 artifact {path.relative_to(root)}"
            )


def _check_hier005_snapshot(
    root: Path,
    snapshot: Any,
    label: str,
    problems: list[str],
    *,
    path_key: str,
    prefix: str = "",
) -> None:
    if not isinstance(snapshot, list) or not snapshot:
        problems.append(f"{label} must be a non-empty list")
        return
    for index, record in enumerate(snapshot):
        item_label = f"{label}[{index}]"
        if not isinstance(record, dict):
            problems.append(f"{item_label} must be an object")
            continue
        raw = record.get(path_key)
        if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
            problems.append(f"{item_label}.{path_key} must be a relative path")
            continue
        descriptor = {
            "path": str(Path(prefix) / raw),
            "sha256": record.get("sha256"),
            "bytes": record.get("bytes"),
        }
        _relative_descriptor_path(root, descriptor, item_label, problems)


def _check_hier005_bundle(
    root: Path,
    manifest: dict[str, Any],
    problems: list[str],
) -> None:
    """Validate the portable, explicitly non-claim HIER diagnostic schemas."""
    schema = manifest.get("schema")
    is_repair = schema == HIER005_REPAIR_SCHEMA
    is_overlap = schema == HIER008_OVERLAP_SCHEMA
    is_dynamic = schema in (
        HIER009_DYNAMIC_SCHEMA,
        HIER010_REFINEMENT_SCHEMA,
        HIER011_EXCHANGE_SCHEMA,
        HIER012_PROJECTION_SCHEMA,
        HIER013_DEVELOPMENT_SCHEMA,
    )
    if manifest.get("status") != "diagnostic":
        problems.append("manifest.json HIER status must be diagnostic")

    records = manifest.get("files")
    observed_paths: set[str] = set()
    if not isinstance(records, list) or not records:
        problems.append("manifest.json HIER files must be a non-empty list")
    else:
        for index, record in enumerate(records):
            label = f"manifest.json files[{index}]"
            path = _relative_descriptor_path(root, record, label, problems)
            raw = record.get("path") if isinstance(record, dict) else None
            if isinstance(raw, str):
                if raw in observed_paths:
                    problems.append(f"{label}: duplicate manifest path {raw!r}")
                observed_paths.add(raw)
            if path is not None and path.name == "manifest.json":
                problems.append(f"{label}: manifest must not recursively describe itself")
        actual_paths = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if observed_paths != actual_paths:
            missing = sorted(actual_paths - observed_paths)
            extra = sorted(observed_paths - actual_paths)
            if missing:
                problems.append(f"manifest.json omits HIER files: {missing!r}")
            if extra:
                problems.append(f"manifest.json names nonexistent HIER files: {extra!r}")

    try:
        config = _load_json(root / "config.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        problems.append(f"config.json is invalid: {exc}")
        config = None
    if not isinstance(config, dict):
        problems.append("config.json must contain an object")
    else:
        if config.get("schema") != schema:
            problems.append("config.json has the wrong HIER diagnostic schema")
        if config.get("status") != "diagnostic" or config.get("claim_ready") is not False:
            problems.append("config.json must remain an explicitly non-claim diagnostic")
        if not isinstance(config.get("command"), str) or not config["command"].strip():
            problems.append("config.json has no executed command")
        if not _finite(config):
            problems.append("config.json contains a non-finite numeric value")
        if is_repair:
            revision = config.get("git_head")
            if not isinstance(revision, str) or not HEX_40.fullmatch(revision):
                problems.append("config.json git_head must be a 40-character Git SHA")
            if not isinstance(config.get("git_dirty"), bool):
                problems.append("config.json git_dirty must be boolean")
            _check_hier005_snapshot(
                root,
                config.get("source_snapshot"),
                "config.json source_snapshot",
                problems,
                path_key="path",
                prefix="source_snapshot",
            )
        else:
            git = config.get("git")
            if not isinstance(git, dict):
                problems.append("config.json git identity must be an object")
            else:
                revision = git.get("revision")
                if not isinstance(revision, str) or not HEX_40.fullmatch(revision):
                    problems.append("config.json git.revision must be a 40-character Git SHA")
                if not isinstance(git.get("branch"), str) or not git["branch"]:
                    problems.append("config.json git.branch must be non-empty")
                if not isinstance(git.get("dirty"), bool):
                    problems.append("config.json git.dirty must be boolean")
            _check_hier005_snapshot(
                root,
                config.get("executed_source_snapshot"),
                "config.json executed_source_snapshot",
                problems,
                path_key="snapshot_path",
            )

    try:
        payload = _load_json(root / "metrics.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        problems.append(f"metrics.json is invalid: {exc}")
        return
    if not isinstance(payload, dict):
        problems.append("metrics.json must contain a HIER diagnostic payload object")
        return
    rows = payload.get("rows")
    if payload.get("schema") != schema:
        problems.append("metrics.json has the wrong HIER diagnostic schema")
    if payload.get("status") != "diagnostic" or payload.get("claim_ready") is not False:
        problems.append("metrics.json must remain an explicitly non-claim diagnostic")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        problems.append("metrics.json rows must be a non-empty object-row list")
        return
    if payload.get("row_count") != len(rows):
        problems.append("metrics.json row_count differs from rows")
    if not _finite(payload):
        problems.append("metrics.json contains a non-finite numeric value")
    _check_json_csv_projection(root, rows, problems)

    linked_paths = _check_html(root, problems)
    required_artifacts: set[Path] = set()
    stable_keys: set[tuple[Any, ...]] = set()
    common_files = {
        "source.png",
        "reconstruction.png",
        "error.png",
        "field.observation.npz",
        "row.json",
    }
    if is_repair:
        required_files = common_files | {
            "source_crop.png",
            "reconstruction_crop.png",
            "error_crop.png",
            "rescue_centers.png",
            "repair_history.json",
        }
    elif is_overlap:
        required_files = common_files | {
            "preoptimization.png",
            "preoptimization_error.png",
            "feature.png",
            "survivors.png",
            "source_crop.png",
            "reconstruction_crop.png",
            "error_crop.png",
            "history.json",
            "optimizer_history.json",
            "analysis.npz",
        }
    elif is_dynamic:
        required_files = common_files | {
            "initial_lattice.png",
            "initial_error.png",
            "feature_priority.png",
            "protected.png",
            "centers.png",
            "source_crop.png",
            "reconstruction_crop.png",
            "error_crop.png",
            "history.json",
            "recovery_history.json",
            "analysis.npz",
            "config.json",
        }
    else:
        required_files = common_files | {"history.json", "recovery_history.json"}
    for index, row in enumerate(rows):
        label = f"metrics.json rows[{index}]"
        if row.get("schema") != schema or row.get("status") != "diagnostic":
            problems.append(f"{label}: wrong schema or non-diagnostic status")
        if is_repair:
            key = (row.get("image"), row.get("rescue_limit"))
        elif is_overlap:
            key = (
                row.get("image"),
                row.get("support_arm"),
                row.get("scheduler"),
                row.get("target_gaussians"),
            )
        elif is_dynamic:
            key = (
                row.get("image"),
                row.get("arm"),
                row.get("target_gaussians"),
            )
        else:
            key = (row.get("image"), row.get("target_gaussians"))
        if key in stable_keys:
            problems.append(f"{label}: duplicate stable row key {key!r}")
        stable_keys.add(key)
        for metric in ("psnr_db", "ms_ssim", "total_seconds"):
            value = row.get(metric)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                problems.append(f"{label}: {metric} must be finite numeric")
        count = row.get("n_gaussians")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            problems.append(f"{label}: n_gaussians must be a positive integer")
        pixel_value = row.get("artifact_pixel_rmse_max")
        pixel_threshold = row.get("artifact_gate_pixel_max_threshold")
        patch_value = row.get("artifact_patch_rmse_max_7")
        patch_threshold = row.get("artifact_gate_patch7_max_threshold")
        gate_values = (pixel_value, pixel_threshold, patch_value, patch_threshold)
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in gate_values
        ):
            problems.append(f"{label}: artifact gate values must be finite numeric")
        else:
            pixel_pass = float(pixel_value) <= float(pixel_threshold)
            patch_pass = float(patch_value) <= float(patch_threshold)
            if row.get("artifact_gate_pixel_max_pass") is not pixel_pass:
                problems.append(f"{label}: pixel gate predicate differs")
            if row.get("artifact_gate_patch7_max_pass") is not patch_pass:
                problems.append(f"{label}: patch gate predicate differs")
            if row.get("artifact_gate_pass") is not (pixel_pass and patch_pass):
                problems.append(f"{label}: combined artifact gate predicate differs")
        parity = row.get("maintained_render_parity_max_abs")
        if (
            not isinstance(parity, (int, float))
            or isinstance(parity, bool)
            or not math.isfinite(float(parity))
            or float(parity) >= 2e-6
        ):
            problems.append(f"{label}: maintained-render parity is missing or too large")
        if is_repair and row.get("base_prefix_bit_exact") is not True:
            problems.append(f"{label}: repair changed the base-field prefix")

        raw_artifact_dir = row.get("artifact_dir")
        if (
            not isinstance(raw_artifact_dir, str)
            or not raw_artifact_dir
            or Path(raw_artifact_dir).is_absolute()
        ):
            problems.append(f"{label}: artifact_dir must be a non-empty relative path")
            continue
        artifact_dir = root / raw_artifact_dir
        try:
            artifact_dir.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (FileNotFoundError, ValueError):
            problems.append(f"{label}: artifact_dir is missing or escapes the bundle")
            continue
        for name in sorted(required_files):
            artifact = artifact_dir / name
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                problems.append(f"{label}: missing required artifact {name}")
            else:
                required_artifacts.add(artifact.resolve())
        field_path = artifact_dir / "field.observation.npz"
        if field_path.is_file() and row.get("field_file_sha256") != _sha256(field_path):
            problems.append(f"{label}: field_file_sha256 differs")
        row_path = artifact_dir / "row.json"
        if row_path.is_file():
            try:
                stored_row = _load_json(row_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                problems.append(f"{label}: row.json is invalid: {exc}")
            else:
                if stored_row != row:
                    problems.append(f"{label}: row.json differs from the metrics ledger")

    try:
        curve_catalog = _load_json(root / "curves" / "catalog.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        problems.append(f"curve catalog is invalid: {exc}")
    else:
        curves = curve_catalog.get("curves") if isinstance(curve_catalog, dict) else None
        if (
            not isinstance(curve_catalog, dict)
            or curve_catalog.get("schema") != schema
            or not isinstance(curves, list)
            or not curves
        ):
            problems.append("curve catalog has the wrong schema or no curves")
        else:
            for index, curve in enumerate(curves):
                raw = curve.get("path") if isinstance(curve, dict) else None
                artifact = _artifact_path(root, raw)
                if artifact is None:
                    problems.append(f"curve catalog[{index}] path is missing or non-portable")
                else:
                    required_artifacts.add(artifact)

    if is_repair:
        try:
            verification = _load_json(root / "verification.json")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            problems.append(f"verification.json is invalid: {exc}")
        else:
            if not isinstance(verification, dict):
                problems.append("verification.json must contain an object")
            elif (
                verification.get("schema") != schema
                or verification.get("status") != "verified_diagnostic"
                or verification.get("metric_rows_checked") != len(rows)
                or verification.get("base_prefix_bit_exact") is not True
            ):
                problems.append("verification.json disagrees with the repair ledger")

    for artifact in sorted(required_artifacts):
        if artifact not in linked_paths:
            problems.append(
                f"index.html does not expose required HIER artifact "
                f"{artifact.relative_to(root)}"
            )


def _check_hier015_plus_bundle(
    root: Path,
    manifest: dict[str, Any],
    problems: list[str],
) -> None:
    """Validate the immutable HIER-015--017 dirty-source diagnostic bundles.

    These source-bound experiments predate the richer HIER-005 report envelope but retain an
    exhaustive file manifest, three exact tabular projections, per-cell row mirrors, source
    snapshots, lossless fields, and portable visual reports.  They are explicitly diagnostic;
    this checker validates integrity and portability, not claim readiness.
    """
    schema = manifest.get("schema")
    if manifest.get("status") != "diagnostic":
        problems.append("manifest.json HIER-015+ status must be diagnostic")

    records = manifest.get("files")
    observed_paths: set[str] = set()
    if not isinstance(records, list) or not records:
        problems.append("manifest.json HIER-015+ files must be a non-empty list")
    else:
        for index, record in enumerate(records):
            label = f"manifest.json files[{index}]"
            _relative_descriptor_path(root, record, label, problems)
            raw = record.get("path") if isinstance(record, dict) else None
            if isinstance(raw, str):
                if raw in observed_paths:
                    problems.append(f"{label}: duplicate manifest path {raw!r}")
                observed_paths.add(raw)
        actual_paths = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if observed_paths != actual_paths:
            missing = sorted(actual_paths - observed_paths)
            extra = sorted(observed_paths - actual_paths)
            if missing:
                problems.append(f"manifest.json omits HIER-015+ files: {missing!r}")
            if extra:
                problems.append(f"manifest.json names nonexistent HIER-015+ files: {extra!r}")

    try:
        config = _load_json(root / "config.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        problems.append(f"config.json is invalid: {exc}")
        config = None
    if not isinstance(config, dict):
        problems.append("config.json must contain an object")
    else:
        if config.get("schema") != schema or config.get("status") != "diagnostic":
            problems.append("config.json has the wrong HIER-015+ schema or status")
        if not isinstance(config.get("command"), str) or not config["command"].strip():
            problems.append("config.json has no executed command")
        if not _finite(config):
            problems.append("config.json contains a non-finite numeric value")
        git = config.get("git")
        if not isinstance(git, dict):
            problems.append("config.json git identity must be an object")
        else:
            revision = git.get("revision")
            if not isinstance(revision, str) or not HEX_40.fullmatch(revision):
                problems.append("config.json git.revision must be a 40-character Git SHA")
            if not isinstance(git.get("branch"), str) or not git["branch"]:
                problems.append("config.json git.branch must be non-empty")
            if not isinstance(git.get("dirty"), bool):
                problems.append("config.json git.dirty must be boolean")
        _check_hier005_snapshot(
            root,
            config.get("source_snapshots"),
            "config.json source_snapshots",
            problems,
            path_key="snapshot_path",
        )

    try:
        payload = _load_json(root / "metrics.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        problems.append(f"metrics.json is invalid: {exc}")
        return
    if not isinstance(payload, dict):
        problems.append("metrics.json must contain a HIER-015+ payload object")
        return
    rows = payload.get("rows")
    if payload.get("schema") != schema or payload.get("status") != "diagnostic":
        problems.append("metrics.json has the wrong HIER-015+ schema or status")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        problems.append("metrics.json rows must be a non-empty object-row list")
        return
    if not _finite(payload):
        problems.append("metrics.json contains a non-finite numeric value")

    jsonl_rows = _load_jsonl(root / "metrics.jsonl", problems)
    if jsonl_rows != rows:
        problems.append("metrics.jsonl rows do not exactly match metrics.json")
    try:
        with (root / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            csv_fields = reader.fieldnames or []
            csv_rows = list(reader)
    except (OSError, csv.Error) as exc:
        problems.append(f"metrics.csv is invalid: {exc}")
        csv_fields, csv_rows = [], []
    expected_fields = sorted({key for row in rows for key in row})
    if csv_fields != expected_fields:
        problems.append("metrics.csv columns do not match the HIER-015+ metrics projection")
    if len(csv_rows) != len(rows):
        problems.append("metrics.csv row count differs from metrics.json")
    elif csv_fields == expected_fields:
        for index, (csv_row, json_row) in enumerate(zip(csv_rows, rows)):
            for field in expected_fields:
                value = json_row.get(field)
                if isinstance(value, (dict, list, tuple)):
                    continue  # legacy writer used Python's stable container projection
                expected = "" if value is None else str(value)
                if csv_row.get(field) != expected:
                    problems.append(
                        f"metrics.csv row {index} field {field!r} differs from metrics.json"
                    )

    for filename, key in (("attempts.json", "attempts"), ("decision.json", None)):
        try:
            record = _load_json(root / filename)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            problems.append(f"{filename} is invalid: {exc}")
            continue
        if not isinstance(record, dict) or record.get("schema") != schema:
            problems.append(f"{filename} has the wrong HIER-015+ schema")
        elif not _finite(record):
            problems.append(f"{filename} contains a non-finite numeric value")
        if key is not None:
            attempts = record.get(key) if isinstance(record, dict) else None
            if not isinstance(attempts, list) or not attempts:
                problems.append(f"{filename} has no attempt ledger")

    linked_paths = _check_html(root, problems)
    stable_keys: set[tuple[Any, Any, Any]] = set()
    for index, row in enumerate(rows):
        label = f"metrics.json rows[{index}]"
        if row.get("schema") != schema or row.get("status") != "diagnostic":
            problems.append(f"{label}: wrong schema or status")
        key = (row.get("image"), row.get("arm"), row.get("target_gaussians"))
        if key in stable_keys:
            problems.append(f"{label}: duplicate stable row key {key!r}")
        stable_keys.add(key)
        for metric in (
            "masked_mse",
            "psnr_db",
            "ms_ssim",
            "lpips",
            "maintained_render_parity_max_abs",
            "repeated_render_parity_max_abs",
        ):
            value = row.get(metric)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                problems.append(f"{label}: {metric} must be finite numeric")
        count = row.get("n_gaussians")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            problems.append(f"{label}: n_gaussians must be a positive integer")
        gaussian_artifact = (
            isinstance(row.get("artifact_dir"), str)
            and (root / str(row["artifact_dir"]) / "field.gaussian.npz").is_file()
        )
        parity_limit = 2e-5 if gaussian_artifact else 1e-3
        for parity_name in (
            "maintained_render_parity_max_abs",
            "repeated_render_parity_max_abs",
        ):
            parity = row.get(parity_name)
            if isinstance(parity, (int, float)) and float(parity) > parity_limit:
                problems.append(f"{label}: {parity_name} exceeds {parity_limit:g}")

        raw_artifact_dir = row.get("artifact_dir")
        if (
            not isinstance(raw_artifact_dir, str)
            or not raw_artifact_dir
            or Path(raw_artifact_dir).is_absolute()
        ):
            problems.append(f"{label}: artifact_dir must be a non-empty relative path")
            continue
        artifact_dir = root / raw_artifact_dir
        try:
            artifact_dir.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (FileNotFoundError, ValueError):
            problems.append(f"{label}: artifact_dir is missing or escapes the bundle")
            continue
        common = {
            "source.png",
            "reconstruction.png",
            "error.png",
            "source_crop.png",
            "reconstruction_crop.png",
            "error_crop.png",
            "analysis.npz",
            "row.json",
            "projection_history.json",
            "geometry_history.json",
        }
        gaussian = (artifact_dir / "field.gaussian.npz").is_file()
        required = common | (
            {"field.gaussian.npz", "fit_history.json"}
            if gaussian
            else {"field.observation.npz"}
        )
        if gaussian and row.get("normalization_eps") is not None:
            required |= {"denominator.json", "denominator.npz"}
        for name in sorted(required):
            artifact = artifact_dir / name
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                problems.append(f"{label}: missing required artifact {name}")
        field_path = artifact_dir / (
            "field.gaussian.npz" if gaussian else "field.observation.npz"
        )
        if field_path.is_file() and row.get("field_file_sha256") != _sha256(field_path):
            problems.append(f"{label}: field_file_sha256 differs")
        row_path = artifact_dir / "row.json"
        if row_path.is_file():
            try:
                stored_row = _load_json(row_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                problems.append(f"{label}: row.json is invalid: {exc}")
            else:
                if stored_row != row:
                    problems.append(f"{label}: row.json differs from the metrics ledger")
        for name in ("source.png", "reconstruction.png", "error.png", "reconstruction_crop.png"):
            artifact = artifact_dir / name
            if artifact.is_file() and artifact.resolve() not in linked_paths:
                problems.append(
                    f"index.html does not expose HIER-015+ artifact "
                    f"{artifact.relative_to(root)}"
                )


def _check_core019_bundle(
    root: Path,
    manifest: dict[str, Any],
    problems: list[str],
    *,
    allow_dirty: bool,
) -> None:
    """Validate CORE-019's multi-view, explicitly non-claim development diagnostic."""

    if manifest.get("status") != "ok":
        problems.append("manifest.json CORE-019 status must be ok")
    if manifest.get("claim_ready") is not False:
        problems.append("manifest.json CORE-019 claim_ready must remain false")
    command = manifest.get("command")
    if not isinstance(command, str) or not command.strip():
        problems.append("manifest.json CORE-019 has no executed command")
    if not _finite(manifest):
        problems.append("manifest.json CORE-019 contains non-finite numeric values")

    required_links: set[Path] = set()

    def descriptor(value: Any, label: str, *, expose: bool = False) -> Path | None:
        path = _relative_descriptor_path(root, value, label, problems)
        if expose and path is not None:
            required_links.add(path)
        return path

    plan_path = descriptor(manifest.get("plan"), "manifest.json plan", expose=True)
    plan = None
    if plan_path is not None:
        try:
            plan = _load_json(plan_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            problems.append(f"plan.json is invalid: {exc}")
    if not isinstance(plan, dict) or plan.get("schema") != "core019.coherent_depth.plan.v1":
        problems.append("plan.json has the wrong CORE-019 schema")
    else:
        if plan.get("command") != command:
            problems.append("plan.json command differs from manifest.json")
        repositories = plan.get("repositories")
        repository = repositories.get("structsplat") if isinstance(repositories, dict) else None
        if not isinstance(repository, dict):
            problems.append("plan.json has no StructSplat repository identity")
        else:
            revision = repository.get("head")
            if not isinstance(revision, str) or not HEX_40.fullmatch(revision):
                problems.append("plan.json StructSplat head must be a 40-character Git SHA")
            if not isinstance(repository.get("branch"), str) or not repository["branch"]:
                problems.append("plan.json StructSplat branch must be non-empty")
            dirty = repository.get("dirty")
            if not isinstance(dirty, bool):
                problems.append("plan.json StructSplat dirty must be boolean")
            elif dirty and not allow_dirty:
                problems.append(
                    "plan.json StructSplat repository was dirty; use --allow-dirty only for "
                    "this explicitly non-claim diagnostic"
                )
        train = plan.get("train_camera_ids")
        report = plan.get("report_camera_ids")
        if not isinstance(train, list) or not train or not isinstance(report, list) or not report:
            problems.append("plan.json must declare non-empty construction/report camera lists")
        elif set(train) & set(report):
            problems.append("plan.json construction and reporting cameras overlap")
        vggt = plan.get("vggt")
        if (
            not isinstance(vggt, dict)
            or vggt.get("checkpoint_bytes") != 5_026_367_224
            or not HEX_64.fullmatch(str(vggt.get("checkpoint_sha256", "")))
        ):
            problems.append("plan.json lacks the pinned CORE-019 checkpoint receipt")

    for key in ("shared_packets", "feature_receipt", "report"):
        descriptor(manifest.get(key), f"manifest.json {key}")
    field = manifest.get("coherent_depth_field")
    if not isinstance(field, dict):
        problems.append("manifest.json lacks coherent-depth-field artifacts")
    else:
        for key in ("arrays", "receipt", "contact_sheet"):
            descriptor(
                field.get(key),
                f"manifest.json coherent_depth_field.{key}",
                expose=key == "contact_sheet",
            )
    metric_artifacts = manifest.get("metrics")
    if not isinstance(metric_artifacts, dict):
        problems.append("manifest.json lacks metric artifact descriptors")
    else:
        for name in ("metrics.json", "metrics.jsonl", "metrics.csv"):
            descriptor(metric_artifacts.get(name), f"manifest.json metrics.{name}")
    plots = manifest.get("plots")
    if not isinstance(plots, dict):
        problems.append("manifest.json lacks plot artifacts")
    else:
        descriptor(plots.get("all_metric_curves"), "manifest.json all_metric_curves")

    try:
        metrics = _load_json(root / "metrics.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        problems.append(f"metrics.json is invalid: {exc}")
        return
    if not isinstance(metrics, list) or len(metrics) != 4 or not all(
        isinstance(row, dict) for row in metrics
    ):
        problems.append("metrics.json must contain exactly four CORE-019 arm rows")
        return
    if not _finite(metrics):
        problems.append("metrics.json contains non-finite numeric values")
    _check_json_csv_projection(root, metrics, problems)
    expected_arms = {
        "interior",
        "posterior_no_reciprocal",
        "vggt_raw_known_ray",
        "vggt_coherent_wse",
    }
    if {row.get("arm") for row in metrics} != expected_arms:
        problems.append("metrics.json does not contain the frozen four CORE-019 arms")
    for index, row in enumerate(metrics):
        label = f"metrics.json[{index}]"
        if row.get("status") != "ok":
            problems.append(f"{label}: arm status must be ok")
        for key in (
            "initial_reporting_psnr",
            "reporting_psnr",
            "reporting_ms_ssim",
            "reporting_lpips",
            "reporting_gradient_mae",
            "pretraining_seconds",
            "training_native_seconds",
            "original_over_packets_plus_model",
        ):
            value = row.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                problems.append(f"{label}: {key} must be finite numeric")
        for key in ("initial_n_gaussians", "final_n_gaussians", "final_model_bytes"):
            value = row.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                problems.append(f"{label}: {key} must be a positive integer")

    records = manifest.get("records")
    record_arms: set[Any] = set()
    packet_sets: set[tuple[Any, ...]] = set()
    if not isinstance(records, list) or len(records) != 4:
        problems.append("manifest.json records must contain exactly four arms")
    else:
        for index, record in enumerate(records):
            label = f"manifest.json records[{index}]"
            if not isinstance(record, dict) or record.get("status") != "ok":
                problems.append(f"{label}: record must be an ok object")
                continue
            arm = record.get("arm")
            if arm in record_arms:
                problems.append(f"{label}: duplicate arm {arm!r}")
            record_arms.add(arm)
            input_record = record.get("input")
            packet_hashes = (
                input_record.get("packet_hashes") if isinstance(input_record, dict) else None
            )
            if not isinstance(packet_hashes, list) or not packet_hashes or not all(
                isinstance(value, str) and HEX_64.fullmatch(value) for value in packet_hashes
            ):
                problems.append(f"{label}: packet hashes are missing or malformed")
            else:
                packet_sets.add(tuple(packet_hashes))
            for key in ("curves", "history"):
                descriptor(record.get(key), f"{label}.{key}", expose=key == "curves")
            models = record.get("models")
            if not isinstance(models, dict):
                problems.append(f"{label}: model artifacts are missing")
            else:
                for key in ("initial_npz", "initial_ply", "final_npz", "final_ply"):
                    descriptor(
                        models.get(key),
                        f"{label}.models.{key}",
                        expose=key == "final_npz",
                    )
            visuals = record.get("visuals")
            if not isinstance(visuals, dict):
                problems.append(f"{label}: visual artifacts are missing")
                continue
            descriptor(
                visuals.get("contact_sheet"), f"{label}.visuals.contact_sheet", expose=True
            )
            view_artifacts = visuals.get("views")
            if not isinstance(view_artifacts, dict) or len(view_artifacts) != 4:
                problems.append(f"{label}: exactly four reporting-view visual sets are required")
                continue
            for view_name, artifacts in view_artifacts.items():
                if not isinstance(artifacts, dict):
                    problems.append(f"{label}: malformed visual set for {view_name}")
                    continue
                for key in ("target", "initial", "final", "error_x4", "alpha", "depth_support"):
                    descriptor(artifacts.get(key), f"{label}.visuals.{view_name}.{key}")
    if record_arms != expected_arms:
        problems.append("manifest.json records do not match the frozen CORE-019 arms")
    if len(packet_sets) != 1:
        problems.append("CORE-019 arms did not reuse one identical packet set")

    linked_paths = _check_html(root, problems)
    for path in sorted(required_links):
        if path not in linked_paths:
            problems.append(f"index.html does not expose CORE-019 artifact {path.relative_to(root)}")


def check_bundle(
    root: Path,
    *,
    allow_dirty: bool = False,
    allow_error_cells: bool = False,
) -> list[str]:
    """Return all bundle-contract problems without mutating the report."""
    root = root.resolve()
    problems: list[str] = []
    if not root.is_dir():
        return [f"report directory does not exist: {root}"]
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            early_manifest = _load_json(manifest_path)
        except (OSError, json.JSONDecodeError, ValueError):
            early_manifest = None
        if (
            isinstance(early_manifest, dict)
            and early_manifest.get("schema") == BENCH020_REPORT_SCHEMA
        ):
            repository_text = str(REPOSITORY_ROOT)
            if repository_text not in sys.path:
                sys.path.insert(0, repository_text)
            from benchmarks.field_semantics_factorial import (  # noqa: PLC0415
                validate_report_bundle,
            )

            return validate_report_bundle(root)
    for name in REQUIRED_FILES:
        required = root / name
        if not required.is_file():
            problems.append(f"missing required report file: {name}")
        elif not _contained(required, root):
            problems.append(f"required report file escapes the bundle: {name}")
    if problems:
        return problems

    try:
        manifest = _load_json(root / "manifest.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        problems.append(f"manifest.json is invalid: {exc}")
        return problems
    if not isinstance(manifest, dict):
        problems.append("manifest.json must contain an object")
        return problems
    if manifest.get("schema") in HIER005_REPORT_SCHEMAS:
        _check_hier005_bundle(root, manifest, problems)
        return problems
    if manifest.get("schema") in HIER015_PLUS_REPORT_SCHEMAS:
        _check_hier015_plus_bundle(root, manifest, problems)
        return problems
    if manifest.get("schema") == CORE019_REPORT_SCHEMA:
        _check_core019_bundle(root, manifest, problems, allow_dirty=allow_dirty)
        return problems
    if manifest.get("schema") == BENCH019_REPORT_SCHEMA:
        _check_bench019_bundle(
            root,
            manifest,
            problems,
            allow_dirty=allow_dirty,
            allow_error_cells=allow_error_cells,
        )
        return problems
    if manifest.get("schema") != "structsplat.current_pipeline.workflow.v1":
        problems.append("manifest.json has the wrong workflow schema")
    if not isinstance(manifest.get("command"), str) or not manifest["command"].strip():
        problems.append("manifest.json has no executed command")
    _check_repository_identity(
        manifest.get("repository"),
        "manifest.json",
        problems,
        allow_dirty=allow_dirty,
    )

    try:
        metrics = _load_json(root / "metrics.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        problems.append(f"metrics.json is invalid: {exc}")
        return problems
    if not isinstance(metrics, list) or not metrics:
        problems.append("metrics.json must contain at least one row")
        return problems
    if not all(isinstance(row, dict) for row in metrics):
        problems.append("every metrics.json row must be an object")
        return problems
    if not _finite(metrics):
        problems.append("metrics.json contains a non-finite numeric value")

    jsonl_rows = _load_jsonl(root / "metrics.jsonl", problems)
    if jsonl_rows != metrics:
        problems.append("metrics.jsonl rows do not exactly match metrics.json")
    try:
        with (root / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            csv_fields = reader.fieldnames or []
            csv_rows = list(reader)
    except (OSError, csv.Error) as exc:
        problems.append(f"metrics.csv is invalid: {exc}")
        csv_fields = []
        csv_rows = []
    expected_fields = sorted(
        {key for row in metrics for key in row if key not in {"curves", "snapshots"}}
    )
    if csv_fields != expected_fields:
        problems.append("metrics.csv columns do not match the canonical metrics.json projection")
    if len(csv_rows) != len(metrics):
        problems.append(
            f"metrics.csv row count {len(csv_rows)} does not match metrics.json "
            f"row count {len(metrics)}"
        )
    elif csv_fields == expected_fields:
        for index, (csv_row, json_row) in enumerate(zip(csv_rows, metrics)):
            for field in expected_fields:
                expected = _csv_value(json_row.get(field))
                if csv_row.get(field) != expected:
                    problems.append(
                        f"metrics.csv row {index} field {field!r} differs from metrics.json"
                    )

    variants = manifest.get("variants")
    seeds = manifest.get("seeds")
    images = manifest.get("images")
    variant_set = (
        {value for value in variants if isinstance(value, str) and value}
        if isinstance(variants, list)
        else set()
    )
    seed_set = (
        {value for value in seeds if isinstance(value, int) and not isinstance(value, bool)}
        if isinstance(seeds, list)
        else set()
    )
    image_rows = images if isinstance(images, list) else []
    image_set = {
        image.get("relative")
        for image in image_rows
        if isinstance(image, dict)
        and isinstance(image.get("relative"), str)
        and image.get("relative")
    }
    if not variant_set:
        problems.append("manifest.json variants must be a non-empty list")
    elif not isinstance(variants, list) or len(variant_set) != len(variants):
        problems.append("manifest.json variants must be unique non-empty strings")
    if not seed_set:
        problems.append("manifest.json seeds must be a non-empty list")
    elif not isinstance(seeds, list) or len(seed_set) != len(seeds):
        problems.append("manifest.json seeds must be unique integers")
    if not image_set:
        problems.append("manifest.json images must declare non-empty relative IDs")
    elif len(image_set) != len(image_rows):
        problems.append("manifest.json images must declare unique non-empty relative IDs")

    cell_keys: set[tuple[Any, ...]] = set()
    manifest_repository = manifest.get("repository")
    required_artifacts: set[Path] = set()
    for index, row in enumerate(metrics):
        label = f"metrics.json[{index}]"
        if row.get("schema") != "structsplat.current_pipeline.metric.v1":
            problems.append(f"{label}: wrong metric schema")
        status = row.get("status")
        if status not in {"ok", "error"}:
            problems.append(f"{label}: status must be 'ok' or 'error'")
        if status == "error" and not allow_error_cells:
            problems.append(
                f"{label}: error cell is not claim-ready; fix/rerun or use "
                "--allow-error-cells for diagnostics"
            )
        if status == "error" and (
            not isinstance(row.get("error"), str) or not row["error"].strip()
        ):
            problems.append(f"{label}: error cell must retain a non-empty diagnostic")
        method = row.get("method")
        variant = row.get("variant")
        seed = row.get("seed")
        source_id = row.get("source_id")
        if not isinstance(method, str) or not method:
            problems.append(f"{label}: method must be a non-empty string")
        if not isinstance(variant, str) or variant not in variant_set:
            problems.append(f"{label}: variant is not declared by manifest.json")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed not in seed_set:
            problems.append(f"{label}: seed is not declared by manifest.json")
        if not isinstance(source_id, str) or source_id not in image_set:
            problems.append(f"{label}: source_id is not declared by manifest.json")
        key = (
            repr(method),
            repr(variant),
            repr(source_id),
            repr(seed),
        )
        if key in cell_keys:
            problems.append(f"{label}: duplicate stable cell key {key!r}")
        cell_keys.add(key)

        if status != "ok":
            continue
        for metric in ("psnr", "ms_ssim", "total_seconds"):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"{label}: {metric} must be numeric")
            elif not math.isfinite(float(value)):
                problems.append(f"{label}: {metric} must be finite")
        resolved: dict[str, Path] = {}
        for field in REQUIRED_OK_ARTIFACTS:
            raw_artifact = row.get(field)
            if isinstance(raw_artifact, str) and Path(raw_artifact).is_absolute():
                problems.append(f"{label}: {field} is a non-portable absolute path")
                continue
            artifact = _artifact_path(root, raw_artifact)
            if artifact is None:
                problems.append(f"{label}: {field} is missing or outside the report bundle")
            else:
                resolved[field] = artifact
                required_artifacts.add(artifact)
        field_path = resolved.get("field_npz")
        if field_path is not None:
            expected_hash = row.get("field_sha256")
            if not isinstance(expected_hash, str) or not HEX_64.fullmatch(expected_hash):
                problems.append(f"{label}: field_sha256 is missing or malformed")
            elif _sha256(field_path) != expected_hash:
                problems.append(f"{label}: field_sha256 does not match field_npz")

        config_path = resolved.get("config_json")
        if config_path is None:
            continue
        try:
            config = _load_json(config_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            problems.append(f"{label}: config_json is invalid: {exc}")
            continue
        if not isinstance(config, dict):
            problems.append(f"{label}: config_json must contain an object")
            continue
        if config.get("schema") != "structsplat.current_pipeline.run.v1":
            problems.append(f"{label}: config_json has the wrong run schema")
        if config.get("repository") != manifest_repository:
            problems.append(f"{label}: config_json repository identity differs from manifest")
        for field in ("method", "variant", "seed"):
            if config.get(field) != row.get(field):
                problems.append(f"{label}: config_json {field} differs from the metric row")
        source = config.get("source")
        if not isinstance(source, dict) or not HEX_64.fullmatch(str(source.get("sha256", ""))):
            problems.append(f"{label}: config_json has no source SHA-256")
        elif source.get("relative") != row.get("source_id"):
            problems.append(f"{label}: config_json source.relative differs from source_id")

        history_path = resolved.get("history_json")
        if history_path is not None:
            try:
                history = _load_json(history_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                problems.append(f"{label}: history_json is invalid: {exc}")
            else:
                if not isinstance(history, dict):
                    problems.append(f"{label}: history_json must contain an object")
                elif not _finite(history):
                    problems.append(f"{label}: history_json contains a non-finite numeric value")

    linked_paths = _check_html(root, problems)
    for artifact in sorted(required_artifacts):
        if artifact not in linked_paths:
            problems.append(
                f"index.html does not expose required run artifact {artifact.relative_to(root)}"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-error-cells", action="store_true")
    args = parser.parse_args()
    problems = check_bundle(
        args.report_dir,
        allow_dirty=args.allow_dirty,
        allow_error_cells=args.allow_error_cells,
    )
    if problems:
        print(f"check_report_bundle: {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"check_report_bundle: OK ({args.report_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
