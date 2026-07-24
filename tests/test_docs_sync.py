"""The structural docs<->code checker stays green (torch-free; runs anywhere)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_docs_sync():
    path = Path(__file__).resolve().parent.parent / "scripts" / "docs_sync.py"
    spec = importlib.util.spec_from_file_location("docs_sync", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docs_sync_reports_no_drift():
    problems = _load_docs_sync().find_problems()
    assert problems == [], "docs_sync drift:\n" + "\n".join(problems)
