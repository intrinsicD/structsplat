#!/usr/bin/env python3
"""Structural docs<->code consistency checker for StructSplat.

Torch-free by design: the source is parsed with :mod:`ast` (never imported), so this runs
anywhere as part of ``scripts/verify.sh`` and CI without pulling torch or CUDA. Every check is
*structural* — it compares registered names / on-disk paths against documentation, so ordinary
prose edits never trip it. See the ``docs-sync`` skill for how to resolve a reported drift.

Checks:
  1. Required top-level docs exist.
  2. Every ``.claude/skills/*/SKILL.md`` is listed by name in ``CLAUDE.md``.
  3. Every ``cli.py`` subcommand (and alias) is documented in ``README.md``.
  4. Every ``init.py`` ``STRATEGIES`` entry appears in ``README.md`` or ``docs/architecture.md``.
  5. Every path-like token in ``CLAUDE.md`` that names a repo location actually exists.
  6. Every ``src/structsplat`` module carries a module docstring.
  7. Every ``docs/*.md`` and ``docs/adr/*.md`` is reachable by name from an entrypoint
     (``CLAUDE.md``, ``README.md``, ``AGENTS.md``) directly or through another reachable doc —
     the reverse of check 5, so a new doc cannot sit in ``docs/`` with no discovery path.

Sibling structural checkers cover the surfaces this one deliberately does not:
``scripts/check_ara.py`` (claim ledger), ``scripts/check_task_policy.py`` (task tree),
``scripts/check_script_layout.py`` (scripts layout), and
``scripts/check_agent_workflow.py`` (agent configuration and generated context).

Exit code is 0 when clean, 1 when any drift is found (with a per-problem report).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "structsplat"

REQUIRED_DOCS = (
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "docs/architecture.md",
    "docs/agent_workflow.md",
    "docs/theory.md",
    "tasks/INDEX.md",
    "tasks/README.md",
    "tasks/TEMPLATE.md",
    "tasks/SESSION-BRIEF.md",
)

# Top-level directories that make a backticked CLAUDE.md token look like a real repo path.
KNOWN_TOP_DIRS = (
    "src",
    "tests",
    "benchmarks",
    "docs",
    "tasks",
    "scripts",
    "deprecated_scripts",
    ".claude",
    ".agents",
    ".github",
    "ara",
    "results",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_required_docs(problems: list[str]) -> None:
    for rel in REQUIRED_DOCS:
        if not (ROOT / rel).is_file():
            problems.append(f"required doc missing: {rel}")


def check_skills_listed(problems: list[str]) -> None:
    skills_dir = ROOT / ".claude" / "skills"
    claude_path = ROOT / "CLAUDE.md"
    if not skills_dir.is_dir() or not claude_path.is_file():
        return
    claude = _read(claude_path)
    for entry in sorted(skills_dir.iterdir()):
        if (entry / "SKILL.md").is_file() and entry.name not in claude:
            problems.append(f"skill '{entry.name}' is not listed in CLAUDE.md")


def _cli_commands() -> set[str]:
    """Subcommand names + aliases from every ``add_parser(...)`` call in cli.py (via AST)."""
    names: set[str] = set()
    tree = ast.parse(_read(SRC / "cli.py"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_parser":
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            names.add(str(node.args[0].value))
        for kw in node.keywords:
            if kw.arg == "aliases" and isinstance(kw.value, (ast.List, ast.Tuple)):
                names.update(str(el.value) for el in kw.value.elts if isinstance(el, ast.Constant))
    return names


def check_cli_documented(problems: list[str]) -> None:
    readme = _read(ROOT / "README.md")
    for name in sorted(_cli_commands()):
        if name not in readme:
            problems.append(f"CLI command '{name}' is not documented in README.md")


def _init_strategies() -> list[str]:
    """The ``STRATEGIES`` tuple/list literal declared in init.py (via AST)."""
    tree = ast.parse(_read(SRC / "init.py"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "STRATEGIES" for t in node.targets):
            continue
        if isinstance(node.value, (ast.Tuple, ast.List)):
            return [str(el.value) for el in node.value.elts if isinstance(el, ast.Constant)]
    return []


def check_strategies_documented(problems: list[str]) -> None:
    corpus = _read(ROOT / "README.md")
    arch = ROOT / "docs" / "architecture.md"
    if arch.is_file():
        corpus += "\n" + _read(arch)
    for strategy in _init_strategies():
        if strategy not in corpus:
            problems.append(
                f"init strategy '{strategy}' is documented in neither README.md nor "
                "docs/architecture.md"
            )


def check_layout_paths(problems: list[str]) -> None:
    claude = _read(ROOT / "CLAUDE.md")
    for token in re.findall(r"`([^`]+)`", claude):
        token = token.strip()
        if " " in token or "/" not in token:
            continue
        top = token.split("/", 1)[0]
        if top not in KNOWN_TOP_DIRS:
            continue
        candidate = token.rstrip("/.")
        if not (ROOT / candidate).exists():
            problems.append(f"path referenced in CLAUDE.md does not exist: {token}")


def check_module_docstrings(problems: list[str]) -> None:
    for py in sorted(SRC.rglob("*.py")):
        if py.name == "__init__.py":
            continue
        try:
            module = ast.parse(_read(py))
        except SyntaxError as exc:  # pragma: no cover - defensive
            problems.append(f"cannot parse {py.relative_to(ROOT)}: {exc}")
            continue
        if ast.get_docstring(module) is None:
            problems.append(f"module missing docstring: {py.relative_to(ROOT)}")


ENTRYPOINTS = ("CLAUDE.md", "README.md", "AGENTS.md")

# Directories that never contain repository documentation references worth scanning.
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "runs",
    "build",
    "dist",
    "external",
    "node_modules",
}
_TEXT_SUFFIXES = {".md", ".py", ".sh", ".yml", ".yaml", ".toml", ".json"}


def _repo_corpus() -> dict[str, str]:
    """Every readable text file in the working tree, keyed by repo-relative path."""
    corpus: dict[str, str] = {}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        try:
            corpus[path.relative_to(ROOT).as_posix()] = path.read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:  # pragma: no cover - unreadable file
            continue
    return corpus


def check_docs_reachable(problems: list[str]) -> None:
    """No doc under docs/ may be undiscoverable. ``check_layout_paths`` is the other direction.

    Three tiers, matching how this repository actually cites documentation:

    * Top-level ``docs/*.md`` are architecture-level docs an agent should find from the guide, so
      they must be reachable from an entrypoint transitively through other reachable docs.
    * ``docs/adr/NNNN-*.md`` are cited by decision number (``ADR-0004``), not filename, so each
      ADR must be cited as ``ADR-NNNN`` somewhere in the tree.
    * Other subtree docs (``docs/research/``, ``docs/prompts/``) are dated session records cited
      from task files and benchmark docs, so each must be referenced by name somewhere.
    """
    docs_dir = ROOT / "docs"
    if not docs_dir.is_dir():
        problems.append("missing docs/ directory")
        return

    # Tier 1: transitive reachability from an entrypoint.
    top_level = {path.name: path for path in sorted(docs_dir.glob("*.md"))}
    reachable: set[str] = set()
    frontier = [_read(ROOT / name) for name in ENTRYPOINTS if (ROOT / name).is_file()]
    while frontier:
        text = frontier.pop()
        for name in sorted(set(top_level) - reachable):
            if name in text:
                reachable.add(name)
                frontier.append(_read(top_level[name]))
    for name in sorted(set(top_level) - reachable):
        problems.append(
            f"docs/{name} is not reachable from {' / '.join(ENTRYPOINTS)} or any doc they reach "
            "(add it to the CLAUDE.md layout section, or link it from a reachable doc)"
        )

    corpus = _repo_corpus()

    # Tier 2a: every ADR must be cited by its decision number.
    for path in sorted((docs_dir / "adr").glob("*.md")):
        match = re.match(r"(\d+)-", path.name)
        if not match:
            problems.append(f"docs/adr/{path.name} does not start with a decision number")
            continue
        token = f"ADR-{match.group(1)}"
        rel = path.relative_to(ROOT).as_posix()
        if not any(token in text for name, text in corpus.items() if name != rel):
            problems.append(
                f"docs/adr/{path.name} is never cited as {token} (an uncited decision is a "
                "decision nobody can follow)"
            )

    # Tier 2b: every other subtree doc must be referenced by name.
    for path in sorted(docs_dir.rglob("*.md")):
        if path.parent == docs_dir or path.parent.name == "adr":
            continue
        rel = path.relative_to(ROOT).as_posix()
        doc_rel = path.relative_to(docs_dir).as_posix()
        if not any(
            (path.name in text or doc_rel in text) for name, text in corpus.items() if name != rel
        ):
            problems.append(f"docs/{doc_rel} is referenced from nowhere in the repository")


CHECKS = (
    check_required_docs,
    check_skills_listed,
    check_cli_documented,
    check_strategies_documented,
    check_layout_paths,
    check_module_docstrings,
    check_docs_reachable,
)


def find_problems() -> list[str]:
    problems: list[str] = []
    for check in CHECKS:
        check(problems)
    return problems


def main() -> int:
    problems = find_problems()
    if problems:
        print("docs_sync: FAIL")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("docs_sync: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
