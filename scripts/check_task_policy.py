#!/usr/bin/env python3
"""Validate the ``tasks/`` tree, dependency graph, and task-review lifecycle.

``tasks/INDEX.md`` is the task outcome authority. This checker keeps every file linked to that
authority, expands compact dependency notation such as ``FIT-005/006/007``, rejects actionable
dependency cycles, and validates the opt-in workflow ratchet used when a task is picked up.

Historical task records are not rewritten to satisfy newer process schemas. The ratchet is:

* every Index row whose disposition is ``in-progress``, ``in-review``, or
  ``revision-required`` must have an ``## Agent workflow`` section;
* once a task has that section, its role/turn fields are always validated; and
* a workflow-governed task moved under ``tasks/done/`` needs a structured terminal Review.

Run: python scripts/check_task_policy.py
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"

TASK_ID = re.compile(r"^[A-Z]+-\d+[A-Z]?$")
TASK_HEADING = re.compile(r"^#\s+([A-Z]+-\d+[A-Z]?)\b")
ROW = re.compile(r"^\|\s*([A-Z]+-\d+[A-Z]?)\s*\|(.*)$")
SECTION = re.compile(r"^##\s+(.*)$")
DEPENDENCY_TOKEN = re.compile(r"\b([A-Z]+)-(\d+[A-Z]?(?:/\d+[A-Z]?)*)")
FIELD = re.compile(
    r"^-\s*(Driver|Reviewer|Turn|Reviewed revision):\s*(.*?)\s*$",
    re.IGNORECASE,
)
FENCE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})")

TASK_META_FILES = frozenset({"INDEX.md", "README.md", "TEMPLATE.md", "SESSION-BRIEF.md"})

# First word of an Active row's Status. Qualifiers may follow the disposition.
STATUS_WORDS = frozenset(
    {
        "todo",
        "partial",
        "in-progress",
        "in-review",
        "revision-required",
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
ACTIONABLE_DISPOSITIONS = frozenset(
    {"todo", "partial", "in-progress", "in-review", "revision-required", "blocked"}
)
CLOSED_DISPOSITIONS = STATUS_WORDS - ACTIONABLE_DISPOSITIONS
WORKFLOW_REQUIRED_DISPOSITIONS = frozenset({"in-progress", "in-review", "revision-required"})
EXPECTED_TURN = {
    "in-progress": "driver",
    "in-review": "reviewer",
    "revision-required": "driver",
    "blocked": "human",
}

REVIEW_VERDICTS = frozenset(
    {
        "Accepted",
        "Accepted with follow-up",
        "Revision required",
        "Rejected",
        "Inconclusive",
        "Provisionally accepted (self-reviewed)",
    }
)
TERMINAL_REVIEW_VERDICTS = REVIEW_VERDICTS - {"Revision required"}
HANDOFF_SECTIONS = (
    "Objective",
    "Changes",
    "Evidence",
    "Assumptions",
    "Uncertainties",
    "Review focus",
    "Protected actions not taken",
    "Recommended next action",
)
REVIEW_SECTIONS = (
    "Verdict",
    "Self-reviewed",
    "Correctness",
    "Evidence quality",
    "Simplicity",
    "Missing cases",
    "Required changes",
    "Optional improvements",
)
PROTOCOL_REVIEW_SECTIONS = (
    "Reviewer",
    "Verdict",
    "Protocol digest",
    "Digest scope",
    "Outcomes accessed",
    "Review focus",
)
REVISION_AUTHORIZATION_SECTIONS = (
    "Authorized by",
    "Additional rounds",
    "Decision",
    "Date",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")

errors: list[str] = []


@dataclass(frozen=True)
class ActiveRow:
    task_id: str
    title: str
    status: str
    depends: str
    lineno: int


@dataclass(frozen=True)
class RetiredRow:
    task_id: str
    title: str
    path: str
    lineno: int


def err(message: str) -> None:
    errors.append(message)


def status_disposition(status: str) -> str:
    """Return the normalized first-word disposition from an Index status."""
    if not status.strip():
        return ""
    return status.split()[0].strip(".,;:").lower()


def dependency_ids(text: str) -> tuple[str, ...]:
    """Expand task/ADR IDs, including compact ``PREFIX-001/002`` notation."""
    found: list[str] = []
    for match in DEPENDENCY_TOKEN.finditer(text):
        prefix = match.group(1)
        for number in match.group(2).split("/"):
            candidate = f"{prefix}-{number}"
            if candidate not in found:
                found.append(candidate)
    return tuple(found)


def _cells(rest: str) -> list[str]:
    return [cell.strip() for cell in rest.split("|")]


def _contained_regular_file(path: Path, directory: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(directory.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return resolved.is_file()


def parse_index_rows(
    problems: list[str] | None = None,
) -> tuple[dict[str, ActiveRow], dict[str, RetiredRow]]:
    """Parse both task-index tables without importing project code."""
    problems = errors if problems is None else problems
    index = TASKS / "INDEX.md"
    if not index.is_file():
        problems.append("tasks/INDEX.md is missing")
        return {}, {}

    active: dict[str, ActiveRow] = {}
    retired: dict[str, RetiredRow] = {}
    section = ""
    for lineno, line in enumerate(index.read_text(encoding="utf-8").splitlines(), start=1):
        heading = SECTION.match(line)
        if heading:
            section = heading.group(1).strip().lower()
            continue
        row = ROW.match(line)
        if not row:
            continue
        task_id = row.group(1)
        cells = _cells(row.group(2))
        if section.startswith("active"):
            if task_id in active:
                problems.append(f"tasks/INDEX.md:{lineno}: duplicate Active row for {task_id}")
                continue
            active[task_id] = ActiveRow(
                task_id=task_id,
                title=cells[0] if cells else "",
                status=cells[1] if len(cells) > 1 else "",
                depends=cells[2] if len(cells) > 2 else "",
                lineno=lineno,
            )
        elif section.startswith("retired"):
            if task_id in retired:
                problems.append(f"tasks/INDEX.md:{lineno}: duplicate Retired row for {task_id}")
                continue
            retired[task_id] = RetiredRow(
                task_id=task_id,
                title=cells[0] if cells else "",
                path=(cells[1] if len(cells) > 1 else "").strip("`"),
                lineno=lineno,
            )

    if not active:
        problems.append("tasks/INDEX.md has no Active Tasks table rows")
    if not retired:
        problems.append("tasks/INDEX.md has no Retired Done Tasks table rows")
    return active, retired


def parse_index() -> tuple[dict[str, str], dict[str, str]]:
    """Compatibility view: ``({active: status}, {retired: path})``."""
    active, retired = parse_index_rows()
    return (
        {task_id: row.status for task_id, row in active.items()},
        {task_id: row.path for task_id, row in retired.items()},
    )


def task_files() -> tuple[dict[str, Path], dict[str, Path]]:
    """Return active and done task files while rejecting duplicate file IDs."""

    def collect(directory: Path) -> dict[str, Path]:
        found: dict[str, Path] = {}
        if not directory.is_dir():
            return found
        for path in sorted(directory.glob("*.md")):
            if path.name in TASK_META_FILES:
                continue
            if not _contained_regular_file(path, directory):
                err(
                    f"{path.relative_to(ROOT)} is not a regular task file contained in "
                    f"{directory.relative_to(ROOT)}"
                )
                continue
            parts = path.name.split("-", 2)
            if len(parts) < 2:
                err(f"{path.relative_to(ROOT)} does not start with a task ID (AREA-NNN-...)")
                continue
            candidate = f"{parts[0]}-{parts[1]}"
            if not TASK_ID.fullmatch(candidate):
                err(f"{path.relative_to(ROOT)} does not start with a well-formed task ID")
                continue
            if candidate in found:
                err(
                    f"task ID {candidate} is claimed by both "
                    f"{found[candidate].relative_to(ROOT)} and {path.relative_to(ROOT)}"
                )
                continue
            found[candidate] = path

            first_heading = next(
                (
                    match.group(1)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if (match := TASK_HEADING.match(line))
                ),
                None,
            )
            if first_heading != candidate:
                err(
                    f"{path.relative_to(ROOT)} heading ID {first_heading!r} does not match "
                    f"filename ID {candidate}"
                )
        return found

    return collect(TASKS), collect(TASKS / "done")


def _visible_markdown_lines(text: str) -> Iterable[str]:
    """Yield Markdown lines outside fenced code blocks."""
    fence_char: str | None = None
    fence_length = 0
    for line in text.splitlines():
        if fence_char is not None:
            stripped = line.lstrip()
            if re.fullmatch(
                rf"{re.escape(fence_char)}{{{fence_length},}}\s*",
                stripped,
            ):
                fence_char = None
                fence_length = 0
            continue
        match = FENCE.match(line)
        if match:
            marker = match.group(1)
            fence_char = marker[0]
            fence_length = len(marker)
            continue
        yield line


def _agent_workflow_body(text: str) -> str | None:
    body: list[str] = []
    collecting = False
    for line in _visible_markdown_lines(text):
        if line.strip() == "## Agent workflow":
            collecting = True
            continue
        if collecting and line.startswith("## "):
            break
        if collecting:
            body.append(line)
    return "\n".join(body) if collecting else None


def _workflow_fields(body: str, source: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in body.splitlines():
        match = FIELD.match(line)
        if not match:
            continue
        name = match.group(1).lower()
        if name in values:
            err(f"{source}: duplicate Agent workflow field {match.group(1)!r}")
            continue
        values[name] = match.group(2).strip()
    return values


def _structured_blocks(
    body: str,
    block_title: str,
    source: str,
) -> list[dict[str, str]]:
    """Parse exact ``###`` blocks and their ``####`` fields."""
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in body.splitlines():
        if line.strip() == f"### {block_title}":
            current = []
            blocks.append(current)
            continue
        if current is not None and line.startswith("### "):
            current = None
            continue
        if current is not None:
            current.append(line)
    parsed: list[dict[str, str]] = []
    for block_index, block in enumerate(blocks, start=1):
        nested: dict[str, str] = {}
        heading: str | None = None
        collected: list[str] = []
        for line in block:
            if line.startswith("#### "):
                if heading is not None:
                    nested[heading] = "\n".join(collected).strip()
                heading = line.removeprefix("#### ").strip()
                if heading in nested:
                    err(
                        f"{source}: {block_title} block {block_index} duplicates "
                        f"section {heading!r}"
                    )
                collected = []
            elif heading is not None:
                collected.append(line)
        if heading is not None:
            nested[heading] = "\n".join(collected).strip()
        parsed.append(nested)
    return parsed


def _require_structured_sections(
    block: dict[str, str],
    expected: tuple[str, ...],
    source: str,
    label: str,
) -> None:
    for heading in expected:
        if not block.get(heading, "").strip():
            err(f"{source}: {label} section {heading!r} is missing or empty")


def _validate_review_semantics(
    review: dict[str, str],
    *,
    source: str,
    driver: str,
    reviewer: str,
    terminal: bool,
) -> None:
    verdict = review.get("Verdict", "").strip()
    self_reviewed = review.get("Self-reviewed", "").strip().removesuffix(".")
    allowed = TERMINAL_REVIEW_VERDICTS if terminal else REVIEW_VERDICTS
    if verdict not in allowed:
        err(f"{source}: Review Verdict must be one of {sorted(allowed)}, found {verdict!r}")
    if self_reviewed not in {"Yes", "No"}:
        err(f"{source}: Review Self-reviewed must be 'Yes' or 'No', found {self_reviewed!r}")
        return
    same_reviewer = driver.casefold() == reviewer.casefold()
    if self_reviewed == "No" and same_reviewer:
        err(f"{source}: independent review requires distinct Driver and Reviewer labels")
    if self_reviewed == "No" and verdict == "Provisionally accepted (self-reviewed)":
        err(f"{source}: provisional self-reviewed verdict requires Self-reviewed 'Yes'")
    if self_reviewed == "Yes":
        if not same_reviewer:
            err(f"{source}: self-review requires Reviewer to equal Driver")
        if terminal and verdict != "Provisionally accepted (self-reviewed)":
            err(
                f"{source}: terminal Self-reviewed 'Yes' requires Verdict "
                "'Provisionally accepted (self-reviewed)'"
            )


def _validate_protocol_reviews(
    reviews: list[dict[str, str]],
    *,
    source: str,
    driver: str,
) -> None:
    for index, review in enumerate(reviews, start=1):
        label = f"Protocol review {index}"
        _require_structured_sections(
            review,
            PROTOCOL_REVIEW_SECTIONS,
            source,
            label,
        )
        reviewer = review.get("Reviewer", "").strip()
        if reviewer.casefold() in {"", "pending"}:
            err(f"{source}: {label} requires a distinct Reviewer")
        elif reviewer.casefold() == driver.casefold():
            err(f"{source}: {label} Reviewer must differ from the Driver")
        verdict = review.get("Verdict", "").strip()
        if verdict not in {"Approved", "Rejected"}:
            err(f"{source}: {label} Verdict must be 'Approved' or 'Rejected'")
        digest = review.get("Protocol digest", "").strip()
        if not SHA256.fullmatch(digest):
            err(f"{source}: {label} Protocol digest must be a lowercase SHA-256")
        if review.get("Outcomes accessed", "").strip().removesuffix(".") != "No":
            err(f"{source}: {label} must record Outcomes accessed as 'No'")


def _revision_authorization_rounds(
    blocks: list[dict[str, str]],
    *,
    source: str,
) -> list[int]:
    rounds_by_block: list[int] = []
    for index, block in enumerate(blocks, start=1):
        label = f"Revision authorization {index}"
        _require_structured_sections(
            block,
            REVISION_AUTHORIZATION_SECTIONS,
            source,
            label,
        )
        authorizer = block.get("Authorized by", "").strip()
        if authorizer.casefold() in {"", "pending"}:
            err(f"{source}: {label} requires a stable human/maintainer label")
        raw_rounds = block.get("Additional rounds", "").strip()
        try:
            rounds = int(raw_rounds)
        except ValueError:
            err(f"{source}: {label} Additional rounds must be an integer")
            rounds_by_block.append(0)
            continue
        if rounds not in {1, 2}:
            err(f"{source}: {label} may authorize only 1 or 2 additional rounds")
            rounds_by_block.append(0)
            continue
        rounds_by_block.append(rounds)
    return rounds_by_block


def _remaining_authorized_review_rounds(
    body: str,
    reviews: list[dict[str, str]],
    authorization_rounds: list[int],
    *,
    source: str,
) -> int:
    """Consume human-authorized rounds in document order after escalation."""
    consecutive = 0
    escalated = False
    remaining = 0
    review_index = 0
    authorization_index = 0
    for line in body.splitlines():
        heading = line.strip()
        if heading == "### Revision authorization":
            rounds = authorization_rounds[authorization_index]
            authorization_index += 1
            if not escalated:
                err(
                    f"{source}: Revision authorization {authorization_index} must follow "
                    "two consecutive Revision required verdicts"
                )
                continue
            remaining += rounds
            continue
        if heading != "### Review":
            continue
        review_index += 1
        review = reviews[review_index - 1]
        if escalated:
            if remaining:
                remaining -= 1
            else:
                err(
                    f"{source}: Review round {review_index} occurred after two consecutive "
                    "Revision required verdicts without prior bounded authorization"
                )
        if review.get("Verdict", "").strip() == "Revision required":
            consecutive += 1
            if consecutive >= 2:
                escalated = True
        elif not escalated:
            consecutive = 0
    return remaining


def validate_agent_workflow(
    task_id: str,
    path: Path,
    *,
    disposition: str,
    done: bool,
) -> None:
    source = path.relative_to(ROOT).as_posix()
    body = _agent_workflow_body(path.read_text(encoding="utf-8"))
    if body is None:
        if not done and disposition in WORKFLOW_REQUIRED_DISPOSITIONS:
            err(
                f"{source}: Index disposition {disposition!r} requires an "
                "`## Agent workflow` section"
            )
        return

    fields = _workflow_fields(body, source)
    required = ("driver", "reviewer", "turn", "reviewed revision")
    for name in required:
        if not fields.get(name):
            err(f"{source}: Agent workflow field {name!r} is missing or empty")
    driver = fields.get("driver", "")
    reviewer = fields.get("reviewer", "")
    turn = fields.get("turn", "").lower()
    revision = fields.get("reviewed revision", "")
    handoffs = _structured_blocks(body, "Handoff", source)
    reviews = _structured_blocks(body, "Review", source)
    protocol_reviews = _structured_blocks(body, "Protocol review", source)
    _validate_protocol_reviews(
        protocol_reviews,
        source=source,
        driver=driver,
    )
    authorization_blocks = _structured_blocks(body, "Revision authorization", source)
    authorization_rounds = _revision_authorization_rounds(
        authorization_blocks,
        source=source,
    )
    remaining_authorized_rounds = _remaining_authorized_review_rounds(
        body,
        reviews,
        authorization_rounds,
        source=source,
    )

    if turn not in {"driver", "reviewer", "human", "none"}:
        err(f"{source}: Agent workflow Turn has invalid value {turn!r}")

    if not done:
        if (
            disposition in WORKFLOW_REQUIRED_DISPOSITIONS | {"blocked"}
            and driver.casefold() == "pending"
        ):
            err(f"{source}: Driver cannot be pending once work has started")
        expected = EXPECTED_TURN.get(disposition)
        if expected and turn != expected:
            err(
                f"{source}: Index disposition {disposition!r} requires Turn "
                f"{expected!r}, found {turn!r}"
            )
        if disposition in CLOSED_DISPOSITIONS:
            err(
                f"{source}: workflow-governed task {task_id} has terminal disposition "
                f"{disposition!r} but still lives in tasks/; move it to tasks/done/"
            )
        if turn == "reviewer":
            if reviewer.casefold() == "pending":
                err(f"{source}: Reviewer cannot be pending when Turn is reviewer")
            if revision.casefold() == "pending":
                err(f"{source}: Reviewed revision cannot be pending when Turn is reviewer")
            if not handoffs:
                err(f"{source}: reviewer handoff requires a structured `### Handoff` block")
            else:
                _require_structured_sections(
                    handoffs[-1], HANDOFF_SECTIONS, source, "latest Handoff"
                )
        if disposition == "revision-required":
            if reviewer.casefold() == "pending":
                err(f"{source}: Reviewer cannot be pending after a review requested revision")
            if not reviews:
                err(f"{source}: revision-required task requires a structured `### Review` block")
            else:
                _require_structured_sections(reviews[-1], REVIEW_SECTIONS, source, "latest Review")
                _validate_review_semantics(
                    reviews[-1],
                    source=source,
                    driver=driver,
                    reviewer=reviewer,
                    terminal=False,
                )
                if reviews[-1].get("Verdict", "").strip() != "Revision required":
                    err(
                        f"{source}: revision-required disposition requires latest "
                        "Review Verdict 'Revision required'"
                    )
            trailing_revisions = 0
            for review in reversed(reviews):
                if review.get("Verdict", "").strip() != "Revision required":
                    break
                trailing_revisions += 1
            if trailing_revisions >= 2 and remaining_authorized_rounds == 0:
                err(
                    f"{source}: two consecutive Revision required verdicts require "
                    "Turn 'human' and a recorded Revision authorization before another round"
                )
        return

    if turn != "none":
        err(f"{source}: retired workflow-governed task requires Turn 'none', found {turn!r}")
    if driver.casefold() == "pending":
        err(f"{source}: retired workflow-governed task has no Driver")
    if reviewer.casefold() == "pending":
        err(f"{source}: retired workflow-governed task has no Reviewer")
    if revision.casefold() == "pending":
        err(f"{source}: retired workflow-governed task has no reviewed revision")

    if not handoffs:
        err(f"{source}: retired workflow-governed task requires a structured `### Handoff` block")
    else:
        _require_structured_sections(handoffs[-1], HANDOFF_SECTIONS, source, "latest Handoff")
    if not reviews:
        err(f"{source}: retired workflow-governed task requires a structured `### Review` block")
        return
    _require_structured_sections(reviews[-1], REVIEW_SECTIONS, source, "latest Review")
    _validate_review_semantics(
        reviews[-1],
        source=source,
        driver=driver,
        reviewer=reviewer,
        terminal=True,
    )


def _validate_dependency_graph(
    active: dict[str, ActiveRow],
    retired: dict[str, RetiredRow],
    known: set[str],
) -> None:
    graph: dict[str, list[str]] = {}
    actionable = {
        task_id
        for task_id, row in active.items()
        if status_disposition(row.status) in ACTIONABLE_DISPOSITIONS
    }

    for task_id, row in active.items():
        for dependency in dependency_ids(row.depends):
            if dependency == task_id:
                err(f"tasks/INDEX.md:{row.lineno}: task {task_id} depends on itself")
                continue
            if dependency.startswith("ADR-"):
                number = dependency.removeprefix("ADR-")
                if not list((ROOT / "docs" / "adr").glob(f"{number}-*.md")):
                    err(
                        f"tasks/INDEX.md:{row.lineno}: {task_id} depends on {dependency} "
                        f"but no docs/adr/{number}-*.md exists"
                    )
                continue
            if dependency not in known:
                err(
                    f"tasks/INDEX.md:{row.lineno}: {task_id} depends on {dependency!r} "
                    "which is not a known task"
                )
                continue
            if task_id in actionable and dependency in actionable:
                graph.setdefault(task_id, []).append(dependency)

    state: dict[str, int] = {}
    stack: list[str] = []
    reported: set[frozenset[str]] = set()

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for neighbor in graph.get(node, []):
            if state.get(neighbor, 0) == 0:
                visit(neighbor)
            elif state.get(neighbor) == 1:
                start = stack.index(neighbor)
                cycle = stack[start:] + [neighbor]
                key = frozenset(cycle)
                if key not in reported:
                    reported.add(key)
                    err("actionable task dependency cycle: " + " -> ".join(cycle))
        stack.pop()
        state[node] = 2

    for node in sorted(graph):
        if state.get(node, 0) == 0:
            visit(node)


def main() -> int:
    errors.clear()
    active, retired = parse_index_rows()
    open_files, done_files = task_files()

    for task_id, row in sorted(active.items()):
        if task_id not in open_files:
            err(
                f"tasks/INDEX.md lists Active task {task_id} but no "
                f"tasks/{task_id}-*.md file exists (retire the row or add the file)"
            )
        disposition = status_disposition(row.status)
        if not row.status:
            err(f"Active task {task_id} has an empty Status cell")
        elif disposition not in STATUS_WORDS:
            err(
                f"Active task {task_id} has status {row.status!r} whose disposition "
                f"{disposition!r} is not one of {sorted(STATUS_WORDS)}"
            )

    for task_id, row in sorted(retired.items()):
        if not row.path:
            err(f"tasks/INDEX.md Retired row {task_id} has no Path cell")
        elif not _contained_regular_file(TASKS / row.path, TASKS / "done"):
            err(
                f"tasks/INDEX.md Retired row {task_id} points at tasks/{row.path} "
                "which is not a regular file contained in tasks/done/"
            )

    for task_id, path in sorted(open_files.items()):
        if task_id not in active:
            err(
                f"{path.relative_to(ROOT)} is not listed in the Active Tasks table of "
                "tasks/INDEX.md (the index is the outcome authority)"
            )
            continue
        validate_agent_workflow(
            task_id,
            path,
            disposition=status_disposition(active[task_id].status),
            done=False,
        )

    for task_id, path in sorted(done_files.items()):
        if task_id not in retired:
            err(
                f"{path.relative_to(ROOT)} is not listed in the Retired Done Tasks table "
                "of tasks/INDEX.md"
            )
        validate_agent_workflow(task_id, path, disposition="", done=True)

    for task_id in sorted(set(active) & set(retired)):
        err(f"task {task_id} appears in both the Active and Retired tables")

    known = set(active) | set(retired) | set(open_files) | set(done_files)
    _validate_dependency_graph(active, retired, known)

    if errors:
        print(f"check_task_policy: {len(errors)} problem(s):", file=sys.stderr)
        for problem in errors:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        f"check_task_policy: OK ({len(active)} active, {len(retired)} retired, "
        f"{len(open_files)} open files, {len(done_files)} done files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
