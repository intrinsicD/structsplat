#!/usr/bin/env python3
"""Structural validator for the ``tasks/`` tree and ``tasks/INDEX.md``.

``tasks/INDEX.md`` calls itself "the current outcome authority", and it earns that: it records
rejected and design-only outcomes with the gates that produced them. Nothing enforced it before
this checker, so the index and the task files could drift apart silently — an active row with no
file, a retired row pointing at a moved path, a task file no table mentions, or a dependency on
an ID that does not exist.

Torch-free by design, like ``scripts/docs_sync.py``.

Checks:
  1. ``tasks/INDEX.md`` exists and has both the Active and Retired tables.
  2. Every Active row's ID has a matching ``tasks/<ID>-*.md`` file.
  3. Every Retired row's ``Path`` resolves under ``tasks/``.
  4. Every ``tasks/*.md`` file (except INDEX) appears in the Active table.
  5. Every ``tasks/done/*.md`` file appears in the Retired table.
  6. No ID appears in both the Active and Retired tables.
  7. Every ID is unique within its table.
  8. Every ``Depends on`` entry that names a concrete task ID resolves to a known task.
  9. Every Active row's Status begins with a known disposition word.

Run: python scripts/check_task_policy.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"

TASK_ID = re.compile(r"^[A-Z]+-\d+[A-Z]?$")
ROW = re.compile(r"^\|\s*([A-Z]+-\d+[A-Z]?)\s*\|(.*)$")
SECTION = re.compile(r"^##\s+(.*)$")

# First word of an Active row's Status. Free-form qualifiers may follow, as in
# "implemented/screened — rejected by the 500-step guard". The disposition itself is fixed so a
# new status word is a deliberate vocabulary change, not a typo.
STATUS_WORDS = frozenset(
    {
        "todo",
        "partial",
        "in-progress",
        "implemented",
        "implemented/screened",
        "implemented/confirmed",
        "design-only",
        "completed",
        "blocked",
        "terminal",
        "repaired",
        "locally",
        "not",
        "superseded",
        "abandoned",
    }
)

errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def parse_index() -> tuple[dict[str, str], dict[str, str]]:
    """Return ({active_id: status}, {retired_id: path}) from tasks/INDEX.md."""
    index = TASKS / "INDEX.md"
    if not index.is_file():
        err("tasks/INDEX.md is missing")
        return {}, {}

    active: dict[str, str] = {}
    retired: dict[str, str] = {}
    section = ""
    for lineno, line in enumerate(index.read_text(encoding="utf-8").splitlines(), start=1):
        heading = SECTION.match(line)
        if heading:
            section = heading.group(1).strip().lower()
            continue
        row = ROW.match(line)
        if not row:
            continue
        task_id, rest = row.group(1), row.group(2)
        cells = [cell.strip() for cell in rest.split("|")]
        if section.startswith("active"):
            if task_id in active:
                err(f"tasks/INDEX.md:{lineno}: duplicate Active row for {task_id}")
            active[task_id] = cells[1] if len(cells) > 1 else ""
        elif section.startswith("retired"):
            if task_id in retired:
                err(f"tasks/INDEX.md:{lineno}: duplicate Retired row for {task_id}")
            retired[task_id] = (cells[1] if len(cells) > 1 else "").strip("`")

    if not active:
        err("tasks/INDEX.md has no Active Tasks table rows")
    if not retired:
        err("tasks/INDEX.md has no Retired Done Tasks table rows")
    return active, retired


def task_files() -> tuple[dict[str, Path], dict[str, Path]]:
    """Return ({id: path} for tasks/*.md, {id: path} for tasks/done/*.md)."""
    def collect(directory: Path) -> dict[str, Path]:
        found: dict[str, Path] = {}
        if not directory.is_dir():
            return found
        for path in sorted(directory.glob("*.md")):
            if path.name == "INDEX.md":
                continue
            task_id = path.name.split("-", 2)
            if len(task_id) < 2:
                err(f"{path.relative_to(ROOT)} does not start with a task ID (AREA-NNN-...)")
                continue
            candidate = f"{task_id[0]}-{task_id[1]}"
            if not TASK_ID.match(candidate):
                err(f"{path.relative_to(ROOT)} does not start with a well-formed task ID")
                continue
            found[candidate] = path
        return found

    return collect(TASKS), collect(TASKS / "done")


def main() -> int:
    active, retired = parse_index()
    open_files, done_files = task_files()

    for task_id in sorted(active):
        if task_id not in open_files:
            err(
                f"tasks/INDEX.md lists Active task {task_id} but no tasks/{task_id}-*.md file "
                "exists (retire the row or add the file)"
            )

    for task_id, rel in sorted(retired.items()):
        if not rel:
            err(f"tasks/INDEX.md Retired row {task_id} has no Path cell")
            continue
        if not (TASKS / rel).is_file():
            err(f"tasks/INDEX.md Retired row {task_id} points at tasks/{rel} which does not exist")

    for task_id, path in sorted(open_files.items()):
        if task_id not in active:
            err(
                f"{path.relative_to(ROOT)} is not listed in the Active Tasks table of "
                "tasks/INDEX.md (the index is the outcome authority)"
            )

    for task_id, path in sorted(done_files.items()):
        if task_id not in retired:
            err(
                f"{path.relative_to(ROOT)} is not listed in the Retired Done Tasks table of "
                "tasks/INDEX.md"
            )

    for task_id in sorted(set(active) & set(retired)):
        err(f"task {task_id} appears in both the Active and Retired tables")

    known = set(active) | set(retired) | set(open_files) | set(done_files)
    for task_id, status in sorted(active.items()):
        if not status:
            err(f"Active task {task_id} has an empty Status cell")
            continue
        first = status.split()[0].strip(".,;:").lower()
        if first not in STATUS_WORDS:
            err(
                f"Active task {task_id} has status '{status}' whose disposition '{first}' is not "
                f"one of {sorted(STATUS_WORDS)}"
            )

    index_text = (TASKS / "INDEX.md").read_text(encoding="utf-8") if (TASKS / "INDEX.md").is_file() else ""
    for lineno, line in enumerate(index_text.splitlines(), start=1):
        row = ROW.match(line)
        if not row:
            continue
        cells = [cell.strip() for cell in row.group(2).split("|")]
        if len(cells) < 3:
            continue
        depends = cells[2]
        for prefix, number in re.findall(r"\b([A-Z]+)-(\d+)\b", depends):
            # A dependency may name an ADR (a decision under docs/adr/) rather than a task.
            if prefix == "ADR":
                if not list((ROOT / "docs" / "adr").glob(f"{number}-*.md")):
                    err(
                        f"tasks/INDEX.md:{lineno}: {row.group(1)} depends on ADR-{number} but no "
                        f"docs/adr/{number}-*.md exists"
                    )
                continue
            dep_id = f"{prefix}-{number}"
            if dep_id not in known:
                err(
                    f"tasks/INDEX.md:{lineno}: {row.group(1)} depends on '{dep_id}' which is not "
                    "a known task"
                )

    if errors:
        print(f"check_task_policy: {len(errors)} problem(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(
        f"check_task_policy: OK ({len(active)} active, {len(retired)} retired, "
        f"{len(open_files)} open files, {len(done_files)} done files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
