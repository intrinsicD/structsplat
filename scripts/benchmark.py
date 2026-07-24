#!/usr/bin/env python3
"""Benchmark the current StructSplat pipeline and build a portable report."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structsplat.workflows import main_benchmark  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main_benchmark())
