"""Build a portable dashboard from an explicit set of benchmark reports.

The dashboard is intentionally an index, not a benchmark merger.  Inputs must be named on the
command line (or in a bundle manifest); the module never scans ``results/`` and never infers that
unlisted artifacts are current or comparable.

Preferred report manifest schema::

    {
      "schema_version": 1,
      "title": "168 KiB convergence lane",
      "kind": "matched-policy analogue",
      "lane": "168_kib",
      "description": "...",
      "report_dir": ".",
      "completeness": {
        "expected_cells": 16,
        "completed_cells": 16,
        "error_cells": 0
      },
      "coverage": {"images": 4, "methods": 2, "seeds": 2},
      "storage": {
        "budget_bytes": 172032,
        "bytes_per_gaussian": 32,
        "gaussian_cap": 5376
      },
      "caveats": ["Local analogues; not native codec evidence."],
      "links": [{"label": "Report", "path": "index.html"}]
    }

A directory without ``report_manifest.json`` is also accepted.  In that case the dashboard reads
only that explicit directory's ``config.json`` and ``metrics.json``/``metrics.jsonl``.  Expected
coverage is inferred only for recognized list-valued benchmark axes and is visibly marked as
inferred.

The authored HTML follows the Data Analytics portable-dashboard/Recharts host contract.  Its
semantic cards, table, report details, and same-data SVG chart fallback work without JavaScript.
When the packaged runtime helper is explicitly supplied, it upgrades that fallback in place while
keeping the delivered file self-contained.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote, urlparse


MANIFEST_NAME = "report_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
STORAGE_168_KIB_BYTES = 168 * 1024


@dataclass
class ReportRecord:
    """Normalized dashboard metadata for one explicitly selected benchmark report."""

    title: str
    source: str
    report_dir: Path
    kind: str = "benchmark"
    lane: str = ""
    description: str = ""
    expected_cells: int | None = None
    completed_cells: int = 0
    error_cells: int = 0
    coverage: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)
    protocol_sha256: str | None = None
    declared_status: str | None = None
    inferred: bool = False
    priority: int = 100
    input_order: int = 0

    @property
    def observed_cells(self) -> int:
        return self.completed_cells + self.error_cells

    @property
    def missing_cells(self) -> int | None:
        if self.expected_cells is None:
            return None
        return max(0, self.expected_cells - self.observed_cells)

    @property
    def completion_fraction(self) -> float | None:
        if self.expected_cells is None or self.expected_cells <= 0:
            return None
        return min(1.0, self.observed_cells / self.expected_cells)

    @property
    def success_fraction(self) -> float | None:
        if self.expected_cells is None or self.expected_cells <= 0:
            return None
        return min(1.0, self.completed_cells / self.expected_cells)

    @property
    def status(self) -> str:
        if self.declared_status:
            return self.declared_status
        if self.error_cells:
            return "errors"
        if self.expected_cells is None:
            return "available" if self.completed_cells else "unknown"
        if self.completed_cells == self.expected_cells:
            return "complete"
        return "partial"

    @property
    def is_168_kib_lane(self) -> bool:
        lane = self.lane.lower().replace("-", "_")
        if lane in {"168_kib", "168kb", "168_kb"}:
            return True
        budget = _first_present(
            self.storage,
            "budget_bytes",
            "storage_budget_bytes",
            "bytes",
        )
        try:
            return int(budget) == STORAGE_168_KIB_BYTES
        except (TypeError, ValueError):
            return "168 kib" in self.title.lower() or "168-kib" in self.title.lower()


@dataclass(frozen=True)
class DashboardResult:
    output: Path
    reports: int
    expected_cells: int
    completed_cells: int
    error_cells: int


def _first_present(mapping: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _nonnegative_int(value: Any, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if integer < 0 or integer != value:
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL in {path} line {number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected an object in {path} line {number}")
        rows.append(value)
    return rows


def _rows_from_directory(directory: Path) -> list[dict[str, Any]]:
    metrics_json = directory / "metrics.json"
    metrics_jsonl = directory / "metrics.jsonl"
    if metrics_json.is_file():
        payload = _load_json(metrics_json)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return [row for row in payload["rows"] if isinstance(row, dict)]
    if metrics_jsonl.is_file():
        return _load_jsonl(metrics_jsonl)
    return []


def _resolved_config(directory: Path) -> dict[str, Any]:
    path = directory / "config.json"
    if not path.is_file():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {}
    resolved = payload.get("resolved")
    return resolved if isinstance(resolved, dict) else payload


def _axis_count(config: dict[str, Any], plural: str, singular: str | None = None) -> int | None:
    value = config.get(plural)
    if isinstance(value, (list, tuple)):
        return len(value) or None
    if value is not None:
        return 1
    if singular and config.get(singular) is not None:
        return 1
    return None


def _infer_expected(config: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    """Infer only recognized explicit axes; never use discovered sibling directories."""
    coverage: dict[str, Any] = {}
    counts: list[int] = []
    for plural, singular in (
        ("images", "image"),
        ("max_sides", "max_side"),
        ("budgets", "budget"),
        ("methods", "method"),
        ("seeds", "seed"),
    ):
        count = _axis_count(config, plural, singular)
        if count is not None:
            coverage[plural] = count
            counts.append(count)

    # Iteration is an axis only when the config explicitly supplies a list.  A scalar is the
    # shared horizon and must not multiply expected cell count.
    if isinstance(config.get("iters"), (list, tuple)):
        count = len(config["iters"])
        if count:
            coverage["iteration_horizons"] = count
            counts.append(count)

    # A meaningful inferred matrix needs at least image and two experimental axes.  Otherwise
    # display observed cells and leave expected coverage unknown rather than inventing one.
    if "images" not in coverage or len(counts) < 3:
        return None, coverage
    expected = 1
    for count in counts:
        expected *= count
    return expected, coverage


def _directory_links(directory: Path) -> list[tuple[str, str]]:
    candidates = (
        ("Open report", "index.html"),
        ("Summary", "summary.md"),
        ("Metrics CSV", "metrics.csv"),
        ("Metrics JSON", "metrics.json"),
        ("Configuration", "config.json"),
    )
    return [(label, name) for label, name in candidates if (directory / name).is_file()]


def _human_title(path: Path) -> str:
    return path.name.replace("_", " ").replace("-", " ").strip().title() or "Benchmark"


def _report_from_directory(directory: Path, input_order: int) -> ReportRecord:
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"report directory does not exist: {directory}")
    manifest = directory / MANIFEST_NAME
    if manifest.is_file():
        return _report_from_payload(_load_json(manifest), manifest, input_order)

    rows = _rows_from_directory(directory)
    completed = sum(row.get("status", "ok") == "ok" for row in rows)
    errors = len(rows) - completed
    config = _resolved_config(directory)
    expected, coverage = _infer_expected(config)
    caveats = [
        "Coverage was inferred from this explicitly selected directory because it has no "
        "report_manifest.json."
    ]
    if expected is None:
        caveats.append(
            "Expected cell count is unknown; observed rows are not proof of completeness."
        )
    storage: dict[str, Any] = {}
    for key in (
        "storage_budget_bytes",
        "storage_bytes_per_gaussian",
        "storage_gaussian_cap",
        "storage_layout",
    ):
        if config.get(key) is not None:
            storage[key.removeprefix("storage_")] = config[key]
    return _validated_report(
        ReportRecord(
            title=_human_title(directory),
            source=str(directory),
            report_dir=directory,
            kind=str(config.get("benchmark_kind", "benchmark directory")),
            lane=str(config.get("lane", "")),
            description=str(config.get("description", "")),
            expected_cells=expected,
            completed_cells=completed,
            error_cells=errors,
            coverage=coverage,
            storage=storage,
            caveats=caveats,
            links=_directory_links(directory),
            protocol_sha256=config.get("run_protocol_sha256"),
            inferred=True,
            input_order=input_order,
        )
    )


def _normalize_links(payload: Any) -> list[tuple[str, str]]:
    if isinstance(payload, dict):
        return [(str(label), str(path)) for label, path in payload.items()]
    links: list[tuple[str, str]] = []
    for item in _as_list(payload):
        if isinstance(item, str):
            links.append((Path(item).name or "Open", item))
        elif isinstance(item, dict):
            path = _first_present(item, "path", "href", "url")
            if path is None:
                continue
            links.append((str(item.get("label", "Open")), str(path)))
    return links


def _report_from_payload(payload: Any, source: Path, input_order: int) -> ReportRecord:
    if not isinstance(payload, dict):
        raise ValueError(f"report manifest must be an object: {source}")
    version = payload.get("schema_version", MANIFEST_SCHEMA_VERSION)
    if int(version) != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported report manifest schema {version!r}: {source}")

    base = source.parent.resolve()
    report_dir_raw = payload.get("report_dir", ".")
    report_dir = Path(report_dir_raw)
    if not report_dir.is_absolute():
        report_dir = (base / report_dir).resolve()

    completeness = payload.get("completeness", {})
    completeness = completeness if isinstance(completeness, dict) else {}
    expected = _first_present(
        completeness,
        "expected_cells",
        "expected",
        default=_first_present(payload, "expected_cells", "expected"),
    )
    completed = _first_present(
        completeness,
        "completed_cells",
        "completed",
        default=_first_present(payload, "completed_cells", "completed", default=0),
    )
    errors = _first_present(
        completeness,
        "error_cells",
        "errors",
        default=_first_present(payload, "error_cells", "errors", default=0),
    )
    coverage = payload.get("coverage", {})
    storage = payload.get("storage", {})
    caveats = payload.get("caveats", [])
    links = _normalize_links(payload.get("links", []))

    # Friendly path aliases make a small manifest convenient while retaining explicit links.
    for label, key in (
        ("Open report", "report_path"),
        ("Open report", "index_path"),
        ("Summary", "summary_path"),
        ("Metrics", "metrics_path"),
        ("Configuration", "config_path"),
    ):
        if payload.get(key) is not None:
            links.append((label, str(payload[key])))
    if not links:
        links = _directory_links(report_dir)

    deduped_links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, target in links:
        if target in seen:
            continue
        seen.add(target)
        deduped_links.append((label, target))

    record = ReportRecord(
        title=str(payload.get("title", _human_title(report_dir))),
        source=str(source.resolve()),
        report_dir=report_dir,
        kind=str(payload.get("kind", "benchmark")),
        lane=str(payload.get("lane", "")),
        description=str(payload.get("description", "")),
        expected_cells=_nonnegative_int(expected, "expected_cells", allow_none=True),
        completed_cells=int(_nonnegative_int(completed, "completed_cells")),
        error_cells=int(_nonnegative_int(errors, "error_cells")),
        coverage=dict(coverage) if isinstance(coverage, dict) else {},
        storage=dict(storage) if isinstance(storage, dict) else {},
        caveats=[str(value) for value in _as_list(caveats) if str(value).strip()],
        links=deduped_links,
        protocol_sha256=_first_present(payload, "protocol_sha256", "run_protocol_sha256"),
        declared_status=str(payload["status"]) if payload.get("status") else None,
        inferred=False,
        priority=int(payload.get("priority", 100)),
        input_order=input_order,
    )
    return _validated_report(record)


def _validated_report(record: ReportRecord) -> ReportRecord:
    if not record.title.strip():
        raise ValueError(f"report title is empty: {record.source}")
    if record.expected_cells is not None and record.observed_cells > record.expected_cells:
        raise ValueError(
            f"observed cells exceed expected cells for {record.title!r}: "
            f"{record.observed_cells} > {record.expected_cells}"
        )
    return record


def _bundle_entries(path: Path, payload: dict[str, Any]) -> list[Any] | None:
    entries = payload.get("reports")
    if not isinstance(entries, list):
        return None
    resolved: list[Any] = []
    for entry in entries:
        if isinstance(entry, str):
            item = Path(entry)
            resolved.append(item if item.is_absolute() else path.parent / item)
        elif isinstance(entry, dict):
            resolved.append((entry, path))
        else:
            raise ValueError(f"invalid report bundle entry in {path}: {entry!r}")
    return resolved


def load_reports(specs: Sequence[str | Path]) -> list[ReportRecord]:
    """Load exactly the reports named by ``specs``; no directory discovery occurs."""
    pending: list[Any] = [Path(spec) for spec in specs]
    records: list[ReportRecord] = []
    while pending:
        spec = pending.pop(0)
        order = len(records)
        if isinstance(spec, tuple):
            payload, bundle_path = spec
            records.append(_report_from_payload(payload, bundle_path, order))
            continue
        path = Path(spec)
        if path.is_dir():
            records.append(_report_from_directory(path, order))
            continue
        if not path.is_file():
            raise FileNotFoundError(f"report manifest or directory does not exist: {path}")
        payload = _load_json(path)
        if isinstance(payload, dict):
            entries = _bundle_entries(path, payload)
            if entries is not None:
                pending = entries + pending
                continue
        records.append(_report_from_payload(payload, path, order))
    if not records:
        raise ValueError("at least one explicit report is required")
    return sorted(
        records,
        key=lambda record: (
            0 if record.is_168_kib_lane else 1,
            record.priority,
            record.input_order,
        ),
    )


def _fmt_int(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def _fmt_pct(value: float | None) -> str:
    return "Unknown" if value is None else f"{100.0 * value:.1f}%"


def _coverage_summary(record: ReportRecord) -> str:
    parts: list[str] = []
    for key, value in record.coverage.items():
        label = key.replace("_", " ")
        if isinstance(value, (list, tuple)):
            parts.append(f"{len(value)} {label}")
        elif isinstance(value, dict):
            parts.append(f"{label}: {len(value)}")
        else:
            parts.append(f"{label}: {value}")
    return " · ".join(parts) if parts else "Not declared"


def _storage_summary(record: ReportRecord) -> str:
    if not record.storage:
        return "Not declared"
    parts: list[str] = []
    budget = _first_present(record.storage, "budget_bytes", "storage_budget_bytes", "bytes")
    per_g = _first_present(
        record.storage,
        "bytes_per_gaussian",
        "storage_bytes_per_gaussian",
    )
    cap = _first_present(record.storage, "gaussian_cap", "storage_gaussian_cap")
    if budget is not None:
        try:
            parts.append(f"{int(budget):,} bytes")
        except (TypeError, ValueError):
            parts.append(str(budget))
    if per_g is not None:
        parts.append(f"{per_g} B/G")
    if cap is not None:
        try:
            parts.append(f"{int(cap):,} G")
        except (TypeError, ValueError):
            parts.append(f"{cap} G")
    return " · ".join(parts) if parts else "Declared in manifest"


def _status_class(status: str) -> str:
    return "warn" if status in {"partial", "errors", "unknown", "right_censored"} else ""


def _local_href(target: str, report_dir: Path, output_dir: Path) -> str:
    parsed = urlparse(target)
    if parsed.scheme in {"http", "https"}:
        return target
    path = Path(target)
    if not path.is_absolute():
        path = report_dir / path
    relative = os.path.relpath(path.resolve(), output_dir.resolve())
    return quote(Path(relative).as_posix(), safe="/:%#?=&")


def _svg_fallback(records: Sequence[ReportRecord]) -> tuple[str, int]:
    row_height = 46
    height = max(320, 72 + row_height * len(records))
    label_x = 16
    plot_x = 310
    plot_width = 520
    elements = [
        f'<svg viewBox="0 0 960 {height}" role="img" '
        'aria-label="Observed benchmark coverage by explicit report">',
        '<line x1="310" y1="42" x2="830" y2="42" stroke="var(--grid)"/>',
        '<text x="310" y="27" font-size="12" fill="var(--muted)">0%</text>',
        '<text x="830" y="27" text-anchor="end" font-size="12" '
        'fill="var(--muted)">100% observed</text>',
    ]
    for index, record in enumerate(records):
        y = 62 + index * row_height
        ratio = record.completion_fraction
        width = 0.0 if ratio is None else plot_width * ratio
        color = (
            "var(--warning)"
            if record.error_cells
            else "var(--green)"
            if ratio == 1.0
            else "var(--blue)"
            if ratio is not None
            else "var(--muted)"
        )
        title = record.title if len(record.title) <= 38 else record.title[:35] + "…"
        elements += [
            f'<text x="{label_x}" y="{y + 16}" font-size="12" '
            f'fill="var(--text)">{escape(title)}</text>',
            f'<rect x="{plot_x}" y="{y}" width="{plot_width}" height="22" rx="11" '
            'fill="var(--surface-tertiary)"/>',
            f'<rect x="{plot_x}" y="{y}" width="{width:.2f}" height="22" rx="11" fill="{color}"/>',
            f'<text x="850" y="{y + 16}" font-size="12" fill="var(--secondary)">'
            f"{escape(_fmt_pct(ratio))}</text>",
        ]
    elements.append("</svg>")
    return "".join(elements), height


def _chart_payload(records: Sequence[ReportRecord], height: int) -> dict[str, Any]:
    data = [
        {
            "report": record.title,
            "status": record.status,
            "coverage": record.completion_fraction,
        }
        for record in records
        if record.completion_fraction is not None
    ]
    return {
        "charts": [
            {
                "id": "report-completeness",
                "height": height,
                "type": "bar",
                "dataset": {
                    "id": "report-completeness",
                    "title": "Observed benchmark coverage",
                    "data": data,
                    "chart_spec": {
                        "id": "report-completeness",
                        "dataset": "report-completeness",
                        "title": "Observed benchmark coverage",
                        "type": "bar",
                        "encodings": {
                            "x": {"field": "report", "type": "nominal"},
                            "y": {
                                "field": "coverage",
                                "label": "Observed share",
                                "type": "quantitative",
                            },
                            "color": {"field": "status", "type": "nominal"},
                        },
                        "xAxisTitle": "",
                        "yAxisTitle": "",
                        "valueFormat": "percent",
                        "settings": {
                            "orientation": "horizontal",
                            "groupMode": "grouped",
                        },
                    },
                },
            }
        ]
    }


def _report_table(records: Sequence[ReportRecord], output_dir: Path) -> str:
    rows: list[str] = []
    for record in records:
        primary = record.links[0] if record.links else None
        if primary:
            href = _local_href(primary[1], record.report_dir, output_dir)
            title = f'<a href="{escape(href, quote=True)}">{escape(record.title)}</a>'
        else:
            title = escape(record.title)
        missing = record.missing_cells
        rows.append(
            "<tr>"
            f"<td>{title}</td>"
            f"<td>{escape(record.kind)}</td>"
            f'<td><span class="pill {_status_class(record.status)}">'
            f"{escape(record.status)}</span></td>"
            f"<td>{_fmt_int(record.completed_cells)} / {_fmt_int(record.expected_cells)}</td>"
            f"<td>{_fmt_int(record.error_cells)}</td>"
            f"<td>{_fmt_int(missing)}</td>"
            f"<td>{escape(_coverage_summary(record))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Report</th><th>Kind</th><th>Status</th>"
        "<th>Successful / expected</th><th>Errors</th><th>Missing</th><th>Coverage</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _report_details(records: Sequence[ReportRecord], output_dir: Path) -> str:
    cards: list[str] = []
    for record in records:
        links = []
        for label, target in record.links:
            href = _local_href(target, record.report_dir, output_dir)
            links.append(f'<a href="{escape(href, quote=True)}">{escape(label)}</a>')
        link_text = " · ".join(links) if links else "No local report link declared"
        caveats = (
            "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in record.caveats) + "</ul>"
            if record.caveats
            else "<p>None declared.</p>"
        )
        protocol = (
            f"<p><b>Protocol:</b> <code>{escape(str(record.protocol_sha256))}</code></p>"
            if record.protocol_sha256
            else ""
        )
        lane_badge = '<span class="pill">168 KiB lane</span> ' if record.is_168_kib_lane else ""
        cards.append(
            '<section class="card report-card">'
            '<div class="card-head">'
            f"<h3>{lane_badge}{escape(record.title)}</h3>"
            f"<p>{escape(record.kind)} · source: {escape(record.source)}</p>"
            "</div>"
            '<div class="report-body">'
            f"<p>{escape(record.description) if record.description else 'No description declared.'}</p>"
            f"<p><b>Coverage:</b> {escape(_coverage_summary(record))}</p>"
            f"<p><b>Storage:</b> {escape(_storage_summary(record))}</p>"
            f"<p><b>Files:</b> {link_text}</p>"
            f'{protocol}<div class="detail-caveats"><b>Caveats</b>{caveats}</div>'
            "</div></section>"
        )
    return "".join(cards)


def _author_html(
    records: Sequence[ReportRecord],
    output: Path,
    *,
    title: str,
    generated_at: datetime,
) -> tuple[str, dict[str, Any]]:
    known = [record for record in records if record.expected_cells is not None]
    expected = sum(int(record.expected_cells or 0) for record in known)
    completed = sum(record.completed_cells for record in records)
    errors = sum(record.error_cells for record in records)
    complete_reports = sum(record.status == "complete" for record in records)
    unknown_reports = sum(record.expected_cells is None for record in records)
    svg, chart_height = _svg_fallback(records)
    table = _report_table(records, output.parent)
    details = _report_details(records, output.parent)
    first = records[0]
    lane_note = (
        "The explicitly selected 168 KiB lane is first. "
        f"Its declared storage model is {_storage_summary(first)}."
        if first.is_168_kib_lane
        else "No selected report declares the 168 KiB lane; ordering follows manifest priority."
    )
    generated_text = generated_at.astimezone().strftime("%d %b %Y, %H:%M %Z")
    caveat_items = [
        "This page indexes explicitly selected artifacts; it does not merge incompatible "
        "protocols or treat unlisted result directories as evidence.",
        "A successful cell is status=ok. Errors and missing cells remain visible and do not count "
        "as successful coverage.",
        "Directory-only inputs without a report manifest are labeled inferred; their row count is "
        "not silently promoted to decision-grade completeness.",
    ]
    if unknown_reports:
        caveat_items.append(
            f"{unknown_reports} selected report(s) do not declare a trustworthy expected cell count."
        )
    caveats = " ".join(caveat_items)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>{escape(title)}</title>
  <style>
    :root{{--surface:#fff;--bg:#f7f7f7;--surface-tertiary:#f1f3f5;--text:#0d0d0d;
      --secondary:#5d5d5d;--muted:#858585;--border:rgba(13,13,13,.10);
      --border-strong:rgba(13,13,13,.17);--grid:#e2e5e8;--blue:#0169cc;
      --green:#00692a;--warning:#d94d00;--warning-bg:#fff1e8;--positive-bg:#edfaf2;
      --font:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
    *{{box-sizing:border-box}}html,body{{max-width:100%;overflow-x:clip}}
    body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 var(--font)}}
    a{{color:var(--blue);text-decoration:none}}a:hover{{text-decoration:underline}}
    .shell{{width:min(1180px,calc(100% - 32px));margin:18px auto 56px;overflow:hidden;
      border:1px solid var(--border);border-radius:24px;background:var(--surface);
      box-shadow:0 4px 16px rgba(0,0,0,.05)}}
    .topbar{{display:flex;min-height:48px;align-items:center;justify-content:space-between;
      padding:8px 16px;border-bottom:1px solid var(--border);font-weight:500}}
    .brand{{display:flex;align-items:center;gap:9px}}.mark{{width:18px;height:18px;
      border-radius:6px;background:linear-gradient(145deg,var(--blue),#8046d9)}}
    .meta{{max-width:56%;overflow:hidden;color:var(--muted);font-size:12px;
      text-overflow:ellipsis;white-space:nowrap}}main{{padding:46px}}
    .reading{{width:min(860px,100%);margin:0 auto}}.wide{{width:100%;margin:28px auto 44px}}
    .kicker{{margin-bottom:12px;color:var(--blue);font-size:12px;font-weight:650;
      letter-spacing:.08em;text-transform:uppercase}}h1{{max-width:820px;margin:0;font-size:36px;
      font-weight:520;letter-spacing:-1.7px;line-height:42px}}h2{{margin:0 0 10px;font-size:24px;
      font-weight:520;letter-spacing:-.5px}}h3{{margin:0;font-size:14px;line-height:22px}}
    .deck{{max-width:790px;margin:16px 0 28px;color:var(--secondary);font-size:16px;
      line-height:25px}}.summary{{display:grid;grid-template-columns:145px 1fr;gap:24px;
      margin-bottom:26px;padding:24px 0;border-top:1px solid var(--border-strong);
      border-bottom:1px solid var(--border-strong)}}.summary-label{{color:var(--muted);
      font-size:12px;font-weight:650;letter-spacing:.06em;text-transform:uppercase}}
    .summary-body p{{margin:0;color:var(--secondary)}}.metrics{{display:grid;
      grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:44px}}
    .metric{{min-height:122px;padding:16px 20px;border:1px solid var(--border);
      border-radius:22px}}.metric-label{{color:var(--secondary);font-size:12px}}
    .metric-value{{margin:8px 0 4px;font-size:26px;font-weight:520;
      font-variant-numeric:tabular-nums;letter-spacing:-1px}}.metric-note{{color:var(--muted);
      font-size:11px}}.pill{{display:inline-flex;padding:2px 8px;border-radius:999px;
      background:var(--positive-bg);color:var(--green);font-size:11px;font-weight:650}}
    .pill.warn{{background:var(--warning-bg);color:var(--warning)}}
    .card{{overflow:hidden;border:1px solid var(--border);border-radius:22px;
      background:var(--surface)}}.card-head{{padding:16px 20px 14px;
      border-bottom:1px solid var(--border)}}.card-head p{{margin:2px 0 0;color:var(--muted);
      font-size:12px}}.chart-wrap{{padding:16px 24px 16px 30px;overflow-x:auto}}
    [data-recharts-chart]{{min-height:{chart_height}px}}[data-recharts-live]{{display:none;
      width:100%;min-height:{chart_height}px}}[data-recharts-ready="true"] [data-recharts-fallback]
      {{display:none}}[data-recharts-ready="true"] [data-recharts-live]{{display:block}}
    .chart-fallback{{width:100%;min-height:{chart_height}px}}.chart-fallback svg{{display:block;
      width:100%;height:auto;min-width:720px;min-height:{chart_height}px}}
    .chart-note{{padding:12px 20px 15px;border-top:1px solid var(--border);
      color:var(--muted);font-size:11px}}.table-card{{margin:12px 0 44px}}
    .table-scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:12px}}
    th{{padding:11px 13px;border-bottom:1px solid var(--border);color:var(--muted);
      font-weight:650;text-align:left;white-space:nowrap}}td{{padding:12px 13px;
      border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums;
      text-align:left;vertical-align:top;white-space:nowrap}}tbody tr:last-child td{{border-bottom:0}}
    .reports{{display:grid;gap:14px;margin:12px 0 44px}}.report-body{{padding:14px 20px}}
    .report-body p{{margin:4px 0 10px;color:var(--secondary)}}.detail-caveats{{margin-top:12px;
      padding-top:12px;border-top:1px solid var(--border)}}.detail-caveats ul{{margin:6px 0 0;
      padding-left:20px}}code{{font-size:11px;overflow-wrap:anywhere;white-space:normal}}
    .caveat{{margin:36px 0 0;padding:16px 18px;border-radius:14px;
      background:var(--surface-tertiary);color:var(--secondary);font-size:12px}}
    @media(prefers-color-scheme:dark){{:root{{--surface:#212121;--bg:#171717;
      --surface-tertiary:#303030;--text:#fff;--secondary:#d0d0d0;--muted:#aaa;
      --border:rgba(255,255,255,.08);--border-strong:rgba(255,255,255,.17);--grid:#3a3a3a;
      --blue:#48aaff;--green:#72d694;--warning:#ff9e6c;--warning-bg:#4a2206;
      --positive-bg:#143b22}}}}
    @media(max-width:800px){{.shell{{width:100%;margin:0;border:0;border-radius:0}}
      main{{padding:30px 18px}}.metrics{{grid-template-columns:1fr 1fr}}
      .summary{{grid-template-columns:1fr;gap:10px}}}}
    @media(max-width:520px){{h1{{font-size:31px;line-height:37px}}
      .metrics{{grid-template-columns:1fr}}}}
  </style>
</head>
<body>
<div class="shell">
  <header class="topbar"><div class="brand"><span class="mark" aria-hidden="true"></span>
    StructSplat Benchmarks</div><div class="meta">Explicit sources · {escape(generated_text)}</div></header>
  <main data-report-audience="technical">
    <article class="reading">
      <div class="kicker">Benchmark coverage dashboard</div>
      <header><h1>{escape(title)}</h1></header>
      <p class="deck">One portable entry point for explicitly selected benchmark artifacts. 
        Completeness, errors, protocol coverage, storage assumptions, caveats, and source links 
        stay visible without JavaScript.</p>
      <section class="summary"><div class="summary-label">Current readout</div>
        <div class="summary-body"><p><strong>{escape(lane_note)}</strong> The dashboard does not 
        claim that different benchmark protocols are directly comparable.</p></div></section>
      <section class="metrics">
        <div class="metric"><div class="metric-label">Selected reports</div>
          <div class="metric-value">{len(records):,}</div><div class="metric-note">Explicit inputs only</div></div>
        <div class="metric"><div class="metric-label">Successful cells</div>
          <div class="metric-value">{completed:,}</div><div class="metric-note">status=ok</div></div>
        <div class="metric"><div class="metric-label">Known expected cells</div>
          <div class="metric-value">{expected:,}</div><div class="metric-note">Declared or labeled inferred</div></div>
        <div class="metric"><div class="metric-label">Complete reports</div>
          <div class="metric-value">{complete_reports:,}</div><div class="metric-note">{errors:,} error cells visible</div></div>
      </section>
      <section><h2>Coverage before conclusions</h2><p class="deck">Observed coverage includes 
        successful and error cells. The table separates them so a failed cell cannot look complete.</p></section>
    </article>
    <div class="wide"><figure class="card"><div class="card-head"><h3>Observed benchmark coverage</h3>
      <p>Successful plus error cells divided by expected cells; unknown expectations remain unknown.</p></div>
      <div class="chart-wrap"><div data-recharts-chart="report-completeness">
        <div class="chart-fallback" data-recharts-fallback>{svg}</div>
        <div data-recharts-live aria-hidden="true"></div></div></div>
      <figcaption class="chart-note">Static fallback and optional Recharts upgrade use the same 
        normalized manifest rows.</figcaption></figure></div>
    <article class="reading"><section class="card table-card"><div class="card-head">
      <h3>Selected report matrix</h3><p>Links are relative to this dashboard for portable local use.</p>
      </div><div class="table-scroll">{table}</div></section>
      <section><h2>Report details and caveats</h2><div class="reports">{details}</div></section>
      <section class="caveat"><strong>Methodology and limitations.</strong> {escape(caveats)}</section>
    </article>
  </main>
</div>
<noscript><p hidden>The static dashboard is complete without JavaScript.</p></noscript>
<!-- DATA_ANALYTICS_HTML_REPORT_RUNTIME -->
</body>
</html>
"""
    return html, _chart_payload(records, chart_height)


def build_results_index(
    report_specs: Sequence[str | Path],
    output: str | Path = "results/index.html",
    *,
    title: str = "StructSplat benchmark index",
    runtime_helper: str | Path | None = None,
    generated_at: datetime | None = None,
) -> DashboardResult:
    """Generate the dashboard and return its reconciled headline counts."""
    records = load_reports(report_specs)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or datetime.now().astimezone()
    authored, payload = _author_html(records, output, title=title, generated_at=generated_at)

    if runtime_helper is None:
        output.write_text(authored, encoding="utf-8")
    else:
        helper = Path(runtime_helper).resolve()
        if not helper.is_file():
            raise FileNotFoundError(f"Recharts embedding helper does not exist: {helper}")
        with tempfile.TemporaryDirectory(prefix="structsplat-results-index-") as tmp:
            tmpdir = Path(tmp)
            shell_path = tmpdir / "index-shell.html"
            payload_path = tmpdir / "payload.json"
            shell_path.write_text(authored, encoding="utf-8")
            payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            subprocess.run(
                [
                    "python3",
                    str(helper),
                    "--input",
                    str(shell_path),
                    "--payload",
                    str(payload_path),
                    "--output",
                    str(output),
                ],
                check=True,
            )

    return DashboardResult(
        output=output,
        reports=len(records),
        expected_cells=sum(int(record.expected_cells or 0) for record in records),
        completed_cells=sum(record.completed_cells for record in records),
        error_cells=sum(record.error_cells for record in records),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build results/index.html from explicit benchmark manifests/directories"
    )
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="explicit report_manifest.json, bundle manifest, or result directory",
    )
    parser.add_argument("--output", type=Path, default=Path("results/index.html"))
    parser.add_argument("--title", default="StructSplat benchmark index")
    parser.add_argument(
        "--runtime-helper",
        type=Path,
        default=None,
        help="optional packaged Data Analytics embed_html_report_runtime.py helper",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build_results_index(
        args.reports,
        args.output,
        title=args.title,
        runtime_helper=args.runtime_helper,
    )
    print(
        f"wrote {result.output} from {result.reports} explicit reports "
        f"({result.completed_cells}/{result.expected_cells} known cells, "
        f"{result.error_cells} errors)"
    )


if __name__ == "__main__":
    main()
