#!/usr/bin/env python3
"""Thin CLI wrapper for the BENCH-020 sealed semantics factorial."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.field_semantics_factorial import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
