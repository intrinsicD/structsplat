from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmarks.results_index import build_results_index, load_reports


GENERATED_AT = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _write_manifest(directory: Path, payload: dict) -> Path:
    directory.mkdir(parents=True)
    (directory / "index.html").write_text("<p>report</p>\n", encoding="utf-8")
    path = directory / "report_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dashboard_uses_only_explicit_reports_and_puts_168_kib_first(tmp_path: Path):
    results = tmp_path / "results"
    ordinary = _write_manifest(
        results / "ordinary",
        {
            "schema_version": 1,
            "title": "Ordinary bounded run",
            "kind": "native reference",
            "completeness": {
                "expected_cells": 10,
                "completed_cells": 8,
                "error_cells": 1,
            },
            "coverage": {"images": 5, "methods": 1, "seeds": 2},
            "caveats": ["One cell failed."],
            "links": [{"label": "Open report", "path": "index.html"}],
        },
    )
    lane = _write_manifest(
        results / "lane_168",
        {
            "schema_version": 1,
            "title": "168 KiB convergence lane",
            "kind": "matched-policy analogue",
            "lane": "168_kib",
            "description": "Shared representation and convergence protocol.",
            "completeness": {
                "expected_cells": 16,
                "completed_cells": 16,
                "error_cells": 0,
            },
            "coverage": {"images": 4, "methods": 2, "seeds": 2},
            "storage": {
                "budget_bytes": 172032,
                "bytes_per_gaussian": 32,
                "gaussian_cap": 5376,
            },
            "links": [{"label": "Open report", "path": "index.html"}],
        },
    )
    _write_manifest(
        results / "unlisted",
        {
            "schema_version": 1,
            "title": "Must not be discovered",
            "completeness": {"expected_cells": 1, "completed_cells": 1},
        },
    )

    output = results / "index.html"
    result = build_results_index(
        [ordinary, lane],
        output,
        generated_at=GENERATED_AT,
    )
    html = output.read_text(encoding="utf-8")

    assert result.reports == 2
    assert result.expected_cells == 26
    assert result.completed_cells == 24
    assert result.error_cells == 1
    assert html.index("168 KiB convergence lane") < html.index("Ordinary bounded run")
    assert "Must not be discovered" not in html
    assert "172,032 bytes" in html
    assert "32 B/G" in html
    assert "5,376 G" in html
    assert 'href="lane_168/index.html"' in html
    assert 'href="ordinary/index.html"' in html
    assert 'data-recharts-chart="report-completeness"' in html
    assert "data-recharts-fallback" in html
    assert "DATA_ANALYTICS_HTML_REPORT_RUNTIME" in html
    assert "Successful / expected" in html
    assert "One cell failed." in html
    assert "<script src=" not in html
    assert "https://cdn" not in html


def test_explicit_directory_fallback_infers_recognized_matrix_and_links(tmp_path: Path):
    report = tmp_path / "fair_run"
    report.mkdir()
    config = {
        "resolved": {
            "images": ["a.png", "b.png"],
            "budgets": [100],
            "methods": ["left", "right"],
            "seeds": [0, 1],
            "iters": 500,
            "storage_budget_bytes": 172032,
            "storage_bytes_per_gaussian": 32,
            "storage_gaussian_cap": 5376,
        }
    }
    (report / "config.json").write_text(json.dumps(config), encoding="utf-8")
    rows = [{"status": "ok", "method": "left"} for _ in range(7)]
    rows.append({"status": "error", "method": "right", "error": "boom"})
    (report / "metrics.json").write_text(json.dumps(rows), encoding="utf-8")
    (report / "summary.md").write_text("# Summary\n", encoding="utf-8")

    records = load_reports([report])
    assert len(records) == 1
    record = records[0]
    assert record.expected_cells == 8
    assert record.completed_cells == 7
    assert record.error_cells == 1
    assert record.status == "errors"
    assert record.inferred is True
    assert record.is_168_kib_lane is True

    output = tmp_path / "index.html"
    build_results_index([report], output, generated_at=GENERATED_AT)
    html = output.read_text(encoding="utf-8")
    assert "Coverage was inferred" in html
    assert "images: 2" in html
    assert "methods: 2" in html
    assert 'href="fair_run/summary.md"' in html
    assert "7 / 8" in html


def test_bundle_manifest_is_explicit_and_invalid_counts_fail_closed(tmp_path: Path):
    first = _write_manifest(
        tmp_path / "first",
        {
            "schema_version": 1,
            "title": "First",
            "completeness": {"expected_cells": 2, "completed_cells": 2},
        },
    )
    second = _write_manifest(
        tmp_path / "second",
        {
            "schema_version": 1,
            "title": "Second",
            "completeness": {"expected_cells": 3, "completed_cells": 2},
        },
    )
    bundle = tmp_path / "dashboard_manifest.json"
    bundle.write_text(
        json.dumps({"reports": [str(first), str(second)]}),
        encoding="utf-8",
    )

    assert [record.title for record in load_reports([bundle])] == ["First", "Second"]

    invalid = _write_manifest(
        tmp_path / "invalid",
        {
            "schema_version": 1,
            "title": "Invalid",
            "completeness": {
                "expected_cells": 2,
                "completed_cells": 2,
                "error_cells": 1,
            },
        },
    )
    with pytest.raises(ValueError, match="observed cells exceed expected"):
        load_reports([invalid])
