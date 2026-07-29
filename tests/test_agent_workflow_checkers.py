"""The structural agent-workflow checkers stay green (torch-free; run anywhere).

``tests/test_docs_sync.py`` covers ``docs_sync.py``. This module covers its four repository-wide
siblings — the claim ledger, task tree, scripts layout, and agent-workflow drift gate — plus the
skill-naming invariant that keeps this repository's skills from colliding with a sibling
repository's.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _load(name: str) -> ModuleType:
    path = SCRIPTS / f"{name}.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"_checker_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "checker",
    ["check_ara", "check_task_policy", "check_script_layout", "check_agent_workflow"],
)
def test_repository_passes_checker(checker: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert _load(checker).main() == 0, capsys.readouterr().err


def test_every_skill_name_is_prefixed() -> None:
    """An unprefixed skill name collides with a sibling repository's skill of the same name."""
    skills_dir = REPO / ".claude" / "skills"
    skills = sorted(p for p in skills_dir.iterdir() if (p / "SKILL.md").is_file())
    assert skills, "no skills found"
    for skill in skills:
        lines = (skill / "SKILL.md").read_text(encoding="utf-8").splitlines()
        names = [line for line in lines if line.startswith("name:")]
        assert names, f"{skill.name}/SKILL.md has no 'name:' frontmatter field"
        declared = names[0].removeprefix("name:").strip()
        assert declared == skill.name, (
            f"{skill.name}/SKILL.md declares mismatched name {declared!r}"
        )
        assert declared.startswith("structsplat-"), (
            f"skill {declared!r} is not repo-prefixed; an unprefixed name shadows or is shadowed "
            "by a sibling repository's skill"
        )


def test_agents_mirror_covers_every_skill() -> None:
    """.agents/skills must expose every skill, not a stale subset."""
    claude_skills = {
        p.name for p in (REPO / ".claude" / "skills").iterdir() if (p / "SKILL.md").is_file()
    }
    agents_dir = REPO / ".agents" / "skills"
    mirrored = {p.name for p in agents_dir.iterdir() if (p / "SKILL.md").is_file()}
    assert claude_skills == mirrored, (
        f"missing from .agents/skills: {sorted(claude_skills - mirrored)}; "
        f"stale in .agents/skills: {sorted(mirrored - claude_skills)}"
    )


def test_legacy_prose_claims_are_not_extended() -> None:
    """The claim-schema ratchet only tightens: no new IDs may join the legacy allowlist."""
    check_ara = _load("check_ara")
    assert check_ara.LEGACY_PROSE_CLAIMS == frozenset(f"C{n:02d}" for n in range(1, 12)), (
        "LEGACY_PROSE_CLAIMS changed; claims from C12 on must use the nine-field schema"
    )


def test_task_index_has_no_orphan_claim_proof_paths() -> None:
    """Every claim Proof path that looks like a repo path must resolve (regression guard)."""
    check_ara = _load("check_ara")
    claims = check_ara.parse_claims()
    assert len(claims) > 40, f"expected the full claim ledger, parsed {len(claims)}"
    assert check_ara.errors == [], check_ara.errors
