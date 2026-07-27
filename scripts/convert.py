#!/usr/bin/env python3
"""The sole supported CLI for current-best masked/unmasked conversion (ADR-0028)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structsplat.workflows import main_convert  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main_convert())
