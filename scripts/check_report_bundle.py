#!/usr/bin/env python3
"""Validate a portable report produced by StructSplat's maintained workflows.

This is the evidence-handoff gate for ``scripts/convert.py``, ``benchmark.py``,
``ablation.py``, and ``stage_search.py``. It checks the shared manifest/metrics contract, clean
source identity, per-cell artifacts, finite metrics, cross-format agreement, and every local
HTML link.

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
