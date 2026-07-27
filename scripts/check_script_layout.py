#!/usr/bin/env python3
"""Keep ``scripts/`` durable and one-off experiment scripts in ``scripts/experiments/``.

Without this gate, task-specific drivers accumulate at the top level of ``scripts/`` until an
agent cannot tell repository tooling from a spent protocol runner. The rule is an allowlist:
every top-level file in ``scripts/`` must be declared durable here, with a reason. Anything else
belongs in ``scripts/experiments/`` (see that directory's README).

Torch-free by design, like ``scripts/docs_sync.py``.

Run: python scripts/check_script_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

# name -> why it is durable repository tooling rather than a one-off experiment driver.
DURABLE_SCRIPTS: dict[str, str] = {
    "verify.sh": "the repository verification gate; CI runs the same sequence",
    "docs_sync.py": "structural docs<->code checker, part of verify.sh",
    "check_ara.py": "ara/ claim-ledger structural checker, part of verify.sh",
    "check_task_policy.py": "tasks/ tree and INDEX.md checker, part of verify.sh",
    "check_script_layout.py": "this checker, part of verify.sh",
    "install_skills.sh": "symlinks project skills for global agent discovery",
    "convert.py": "folder conversion with the current pipeline",
    "benchmark.py": "current-pipeline benchmark and portable report",
    "ablation.py": "registered current-pipeline ablation study",
    "stage_search.py": "single-image registered stage-variant search",
}


def main() -> int:
    if not SCRIPTS.is_dir():
        print("check_script_layout: missing scripts/ directory", file=sys.stderr)
        return 1

    errors: list[str] = []

    for path in sorted(SCRIPTS.iterdir()):
        if path.is_dir() or path.name.startswith("."):
            continue
        if path.name not in DURABLE_SCRIPTS:
            errors.append(
                f"scripts/{path.name} is not declared durable. Move it to scripts/experiments/ "
                "(see scripts/experiments/README.md), or add it to DURABLE_SCRIPTS in "
                "scripts/check_script_layout.py with a reason."
            )

    for name in sorted(DURABLE_SCRIPTS):
        if not (SCRIPTS / name).is_file():
            errors.append(
                f"DURABLE_SCRIPTS lists scripts/{name} but that file does not exist "
                "(remove the stale allowlist entry)"
            )

    experiments = SCRIPTS / "experiments"
    if not experiments.is_dir():
        errors.append("missing scripts/experiments/ directory")
    elif not (experiments / "README.md").is_file():
        errors.append("missing scripts/experiments/README.md (the layout policy lives there)")

    deprecated = ROOT / "deprecated_scripts"
    if not deprecated.is_dir():
        errors.append("missing deprecated_scripts/ archive")
    elif not (deprecated / "README.md").is_file():
        errors.append("missing deprecated_scripts/README.md")

    if errors:
        print(f"check_script_layout: {len(errors)} problem(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"check_script_layout: OK ({len(DURABLE_SCRIPTS)} durable scripts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
