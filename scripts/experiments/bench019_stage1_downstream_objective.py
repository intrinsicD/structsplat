#!/usr/bin/env python3
"""BENCH-019 frozen-protocol adapter.

See ``benchmarks.stage1_downstream_objective`` for the lifecycle and exact invocations.  This
task-local wrapper keeps the task discoverable under ``scripts/experiments/`` while all reusable
validation and analysis remains importable and unit-testable.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.stage1_downstream_objective import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
