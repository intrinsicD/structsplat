#!/usr/bin/env python3
"""Search every registered variant of one current-pipeline stage."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structsplat.workflows import main_stage_search  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main_stage_search())
