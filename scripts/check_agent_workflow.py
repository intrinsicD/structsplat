#!/usr/bin/env python3
"""Validate StructSplat's agent-facing configuration as one coherent workflow.

Domain checkers validate docs, claims, tasks, and script placement. This checker validates the
orchestration around them: authority adapters, Claude session setup, skill mirrors, the exact
verification spine, CI invocation, generated session context, and the pull-request contract.

Run: python scripts/check_agent_workflow.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable

import generate_session_brief

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = (
    "AGENTS.md",
    "CLAUDE.md",
    "docs/agent_workflow.md",
    "tasks/README.md",
    "tasks/TEMPLATE.md",
    "tasks/SESSION-BRIEF.md",
    ".claude/settings.json",
    ".claude/hooks/session-start.sh",
    ".github/pull_request_template.md",
    ".github/workflows/ci.yml",
    "scripts/verify.sh",
)

EXPECTED_VERIFY_COMMANDS = (
    '"$STRUCTSPLAT_PYTHON_BIN" -m ruff check src tests scripts benchmarks',
    '"$STRUCTSPLAT_PYTHON_BIN" -m pytest -q -m "not slow and not integration"',
    '"$STRUCTSPLAT_PYTHON_BIN" scripts/docs_sync.py',
    '"$STRUCTSPLAT_PYTHON_BIN" scripts/check_ara.py',
    '"$STRUCTSPLAT_PYTHON_BIN" scripts/check_task_policy.py',
    '"$STRUCTSPLAT_PYTHON_BIN" scripts/check_script_layout.py',
    '"$STRUCTSPLAT_PYTHON_BIN" scripts/check_agent_workflow.py',
)
STRUCTURAL_CI_COMMANDS = (
    "python scripts/docs_sync.py",
    "python scripts/check_ara.py",
    "python scripts/check_task_policy.py",
    "python scripts/check_script_layout.py",
    "python scripts/check_agent_workflow.py",
)
SETTINGS_COMMANDS = (
    "Bash(python scripts/docs_sync.py)",
    "Bash(python scripts/check_ara.py)",
    "Bash(python scripts/check_task_policy.py)",
    "Bash(python scripts/check_script_layout.py)",
    "Bash(python scripts/check_agent_workflow.py)",
    "Bash(python scripts/generate_session_brief.py:*)",
    "Bash(python scripts/check_report_bundle.py:*)",
)
PR_HEADINGS = (
    "## Summary",
    "## Task and scope",
    "## Verification",
    "## Results and claims",
    "## Documentation",
    "## Review and handoff",
    "## Risks and follow-ups",
)
STRUCTURAL_PYTHON_MATRIX = 'python-version: ["3.10", "3.11", "3.12", "3.13"]'


def _strip_shell_comment(line: str) -> str:
    """Strip only a whitespace-delimited trailing shell comment."""
    return re.sub(r"\s+#.*$", "", line).strip()


def _executable_shell_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(_strip_shell_comment(raw))
    return lines


def check_verify_script(root: Path, problems: list[str]) -> None:
    path = root / "scripts" / "verify.sh"
    if not path.is_file():
        return
    if not os.access(path, os.X_OK):
        problems.append("scripts/verify.sh is not executable")
    lines = _executable_shell_lines(path.read_text(encoding="utf-8"))
    positions: list[int] = []
    for command in EXPECTED_VERIFY_COMMANDS:
        occurrences = [index for index, line in enumerate(lines) if line == command]
        if len(occurrences) != 1:
            problems.append(
                f"scripts/verify.sh must execute exactly one `{command}`; found {len(occurrences)}"
            )
        else:
            positions.append(occurrences[0])
    if len(positions) == len(EXPECTED_VERIFY_COMMANDS) and positions != sorted(positions):
        problems.append("scripts/verify.sh verification stages are out of canonical order")


def _skill_directories(path: Path) -> dict[str, Path]:
    if not path.is_dir():
        return {}
    return {
        entry.name: entry
        for entry in path.iterdir()
        if entry.is_dir() and (entry / "SKILL.md").is_file()
    }


def check_skills(root: Path, problems: list[str]) -> None:
    canonical = _skill_directories(root / ".claude" / "skills")
    mirror_root = root / ".agents" / "skills"
    mirrored = _skill_directories(mirror_root)
    if not canonical:
        problems.append(".claude/skills has no discoverable skills")
        return
    mirror_entries = (
        {entry.name for entry in mirror_root.iterdir() if not entry.name.startswith(".")}
        if mirror_root.is_dir()
        else set()
    )
    unexpected_entries = mirror_entries - set(canonical)
    if unexpected_entries:
        problems.append(
            f"skill mirror has stale/unrecognized entries: {sorted(unexpected_entries)}"
        )
    if set(canonical) != set(mirrored):
        problems.append(
            "skill mirror mismatch: "
            f"missing={sorted(set(canonical) - set(mirrored))}, "
            f"stale={sorted(set(mirrored) - set(canonical))}"
        )
    for name, source in sorted(canonical.items()):
        if not name.startswith("structsplat-"):
            problems.append(f"canonical skill {name!r} is not repository-prefixed")
        text = (source / "SKILL.md").read_text(encoding="utf-8")
        declared = next(
            (
                line.removeprefix("name:").strip()
                for line in text.splitlines()
                if line.startswith("name:")
            ),
            "",
        )
        if declared != name:
            problems.append(f"skill directory {name!r} declares frontmatter name {declared!r}")
        mirror = root / ".agents" / "skills" / name
        if not mirror.exists():
            continue
        if not mirror.is_symlink():
            problems.append(f".agents/skills/{name} must be a symlink to the canonical skill")
            continue
        try:
            target = mirror.resolve(strict=True)
        except OSError as exc:
            problems.append(f".agents/skills/{name} is a broken symlink: {exc}")
            continue
        if target != source.resolve():
            problems.append(
                f".agents/skills/{name} resolves to {target}, expected {source.resolve()}"
            )


def check_settings(root: Path, problems: list[str]) -> None:
    path = root / ".claude" / "settings.json"
    if not path.is_file():
        return
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f".claude/settings.json is invalid JSON: {exc}")
        return
    if not isinstance(settings, dict):
        problems.append(".claude/settings.json must contain an object")
        return
    hooks_root = settings.get("hooks", {})
    if not isinstance(hooks_root, dict):
        problems.append(".claude/settings.json hooks must be an object")
        hooks_root = {}
    hooks = hooks_root.get("SessionStart", [])
    if not isinstance(hooks, list):
        problems.append(".claude/settings.json hooks.SessionStart must be a list")
        hooks = []
    commands = [
        hook.get("command")
        for group in hooks
        if isinstance(group, dict)
        for hook in (group.get("hooks", []) if isinstance(group.get("hooks", []), list) else [])
        if isinstance(hook, dict)
    ]
    expected_hook = "$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.sh"
    if commands != [expected_hook]:
        problems.append(
            ".claude/settings.json must configure exactly the canonical SessionStart hook"
        )
    permissions = settings.get("permissions", {})
    if not isinstance(permissions, dict):
        problems.append(".claude/settings.json permissions must be an object")
        permissions = {}
    allow = permissions.get("allow", [])
    if not isinstance(allow, list):
        problems.append(".claude/settings.json permissions.allow must be a list")
        return
    for command in SETTINGS_COMMANDS:
        if command not in allow:
            problems.append(f".claude/settings.json does not allow `{command}`")
    ask = permissions.get("ask", [])
    if not isinstance(ask, list):
        problems.append(".claude/settings.json permissions.ask must be a list")
        ask = []
    for command in ("Bash(git push:*)", "Bash(pip install:*)"):
        if command not in ask:
            problems.append(f".claude/settings.json must ask before `{command}`")

    hook_path = root / ".claude" / "hooks" / "session-start.sh"
    if hook_path.is_file():
        if not os.access(hook_path, os.X_OK):
            problems.append(".claude/hooks/session-start.sh is not executable")
        hook_text = hook_path.read_text(encoding="utf-8")
        if "tasks/SESSION-BRIEF.md" not in hook_text:
            problems.append("SessionStart hook does not expose tasks/SESSION-BRIEF.md")


def check_authority_adapters(root: Path, problems: list[str]) -> None:
    agents_path = root / "AGENTS.md"
    if agents_path.is_file():
        agents = agents_path.read_text(encoding="utf-8")
        for token in (
            "CLAUDE.md",
            "tasks/SESSION-BRIEF.md",
            "docs/agent_workflow.md",
            "./scripts/verify.sh",
        ):
            if token not in agents:
                problems.append(f"AGENTS.md does not point readers to {token}")

    claude_path = root / "CLAUDE.md"
    if claude_path.is_file():
        claude = claude_path.read_text(encoding="utf-8")
        for token in (
            "tasks/SESSION-BRIEF.md",
            "docs/agent_workflow.md",
            "check_agent_workflow.py",
            "check_report_bundle.py",
        ):
            if token not in claude:
                problems.append(f"CLAUDE.md does not document {token}")


def check_session_brief(root: Path, problems: list[str]) -> None:
    path = root / generate_session_brief.OUTPUT
    try:
        expected = generate_session_brief.generate(root)
    except ValueError as exc:
        problems.append(f"cannot generate tasks/SESSION-BRIEF.md: {exc}")
        return
    actual = path.read_text(encoding="utf-8") if path.is_file() else None
    if actual != expected:
        problems.append(
            "tasks/SESSION-BRIEF.md is stale; run `python scripts/generate_session_brief.py`"
        )


def _noncomment_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        yield line


def check_ci(root: Path, problems: list[str]) -> None:
    path = root / ".github" / "workflows" / "ci.yml"
    if not path.is_file():
        return
    text = "\n".join(_noncomment_lines(path.read_text(encoding="utf-8")))
    if len(re.findall(r"^\s*run:\s*\./scripts/verify\.sh\s*$", text, re.MULTILINE)) != 1:
        problems.append("CI must invoke `./scripts/verify.sh` verbatim exactly once")
    structural_match = re.search(
        r"^  structural:\s*\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\s*$|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    structural = structural_match.group("body") if structural_match else ""
    if not structural_match:
        problems.append("CI has no structural job")
    for command in STRUCTURAL_CI_COMMANDS:
        if structural.count(command) != 1:
            problems.append(f"CI structural job must list `{command}` exactly once")
    if not re.search(r"^permissions:\s*\n\s+contents:\s+read\s*$", text, re.MULTILINE):
        problems.append("CI must declare read-only repository contents permission")
    if STRUCTURAL_PYTHON_MATRIX not in structural:
        problems.append(
            "CI structural job must cover the supported Python 3.10-3.13 checker matrix"
        )
    if "python-version: ${{ matrix.python-version }}" not in structural:
        problems.append("CI structural job does not consume its Python-version matrix")


def check_pr_template(root: Path, problems: list[str]) -> None:
    path = root / ".github" / "pull_request_template.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    positions: list[int] = []
    for heading in PR_HEADINGS:
        count = text.count(heading)
        if count != 1:
            problems.append(f"pull request template must contain exactly one `{heading}`")
        else:
            positions.append(text.index(heading))
    if len(positions) == len(PR_HEADINGS) and positions != sorted(positions):
        problems.append("pull request template headings are out of canonical order")


def check(root: Path = ROOT) -> list[str]:
    root = root.resolve()
    problems: list[str] = []
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            problems.append(f"missing required workflow file: {relative}")
    check_authority_adapters(root, problems)
    check_settings(root, problems)
    check_skills(root, problems)
    check_verify_script(root, problems)
    check_session_brief(root, problems)
    check_ci(root, problems)
    check_pr_template(root, problems)
    return problems


def main() -> int:
    problems = check()
    if problems:
        print(f"check_agent_workflow: {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("check_agent_workflow: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
