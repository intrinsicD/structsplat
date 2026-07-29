"""Adversarial tests for task handoff, generated context, and report integrity."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _load(name: str) -> ModuleType:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_maturity_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _workflow_text(
    *,
    driver: str = "driver-a",
    reviewer: str = "reviewer-b",
    turn: str = "none",
    revision: str = "tree-123",
    verdict: str = "Accepted",
    self_reviewed: str = "No",
    include_handoff: bool = True,
    include_review: bool = True,
) -> str:
    handoff = (
        """
### Handoff

#### Objective
Close the bounded task.

#### Changes
Implemented the requested behavior.

#### Evidence
Focused checks passed.

#### Assumptions
None beyond the task.

#### Uncertainties
None known.

#### Review focus
Inspect the lifecycle boundary.

#### Protected actions not taken
No external or destructive action.

#### Recommended next action
Review and dispose the task.
"""
        if include_handoff
        else ""
    )
    review = (
        f"""
### Review

#### Verdict
{verdict}

#### Self-reviewed
{self_reviewed}

#### Correctness
The bounded behavior is correct.

#### Evidence quality
The focused checks exercise the contract.

#### Simplicity
No duplicate authority was added.

#### Missing cases
None material.

#### Required changes
None.

#### Optional improvements
None.
"""
        if include_review
        else ""
    )
    return f"""# DOCS-999 — Fixture

## Agent workflow
- Driver: {driver}
- Reviewer: {reviewer}
- Turn: {turn}
- Reviewed revision: {revision}
{handoff}
{review}
"""


def _validate_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    *,
    disposition: str,
    done: bool,
) -> list[str]:
    checker = _load("check_task_policy")
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    task_dir = tmp_path / "tasks" / ("done" if done else "")
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / "DOCS-999-fixture.md"
    path.write_text(text, encoding="utf-8")
    checker.errors.clear()
    checker.validate_agent_workflow(
        "DOCS-999",
        path,
        disposition=disposition,
        done=done,
    )
    return list(checker.errors)


def test_compact_dependencies_expand_and_actionable_cycles_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _load("check_task_policy")
    assert checker.dependency_ids("FIT-005/006/007, ADR-0029/0030") == (
        "FIT-005",
        "FIT-006",
        "FIT-007",
        "ADR-0029",
        "ADR-0030",
    )

    monkeypatch.setattr(checker, "ROOT", tmp_path)
    checker.errors.clear()
    active = {
        "DOCS-001": checker.ActiveRow("DOCS-001", "one", "todo", "DOCS-002", 1),
        "DOCS-002": checker.ActiveRow("DOCS-002", "two", "partial", "DOCS-001, DOCS-404", 2),
    }
    checker._validate_dependency_graph(active, {}, set(active))
    assert any("actionable task dependency cycle" in error for error in checker.errors)
    assert any("DOCS-404" in error and "not a known task" in error for error in checker.errors)


def test_task_files_and_retired_paths_cannot_escape_by_symlink(tmp_path: Path) -> None:
    checker = _load("check_task_policy")
    done = tmp_path / "tasks" / "done"
    done.mkdir(parents=True)
    internal = done / "DOCS-001-internal.md"
    internal.write_text("# DOCS-001 — internal\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("# DOCS-002 — outside\n", encoding="utf-8")
    link = done / "DOCS-002-link.md"
    link.symlink_to(outside)

    assert checker._contained_regular_file(internal, done)
    assert not checker._contained_regular_file(link, done)
    assert not checker._contained_regular_file(done / "missing.md", done)


def test_terminal_self_review_is_provisional_and_casefolded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disguised = _workflow_text(driver="Codex", reviewer="codex")
    errors = _validate_workflow(
        tmp_path,
        monkeypatch,
        disguised,
        disposition="",
        done=True,
    )
    assert any("independent review requires distinct" in error for error in errors)

    pending = _workflow_text(reviewer="pending")
    errors = _validate_workflow(
        tmp_path,
        monkeypatch,
        pending,
        disposition="",
        done=True,
    )
    assert any("has no Reviewer" in error for error in errors)

    valid = _workflow_text(
        driver="codex",
        reviewer="codex",
        verdict="Provisionally accepted (self-reviewed)",
        self_reviewed="Yes",
    )
    assert (
        _validate_workflow(
            tmp_path,
            monkeypatch,
            valid,
            disposition="",
            done=True,
        )
        == []
    )


def test_review_turn_requires_bound_revision_and_structured_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _workflow_text(
        turn="reviewer",
        revision="pending",
        include_handoff=False,
        include_review=False,
    )
    errors = _validate_workflow(
        tmp_path,
        monkeypatch,
        text,
        disposition="in-review",
        done=False,
    )
    assert any("Reviewed revision cannot be pending" in error for error in errors)
    assert any("structured `### Handoff`" in error for error in errors)


def test_prospective_protocol_review_requires_independence_digest_and_sealed_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _workflow_text().replace(
        "\n### Handoff\n",
        """
### Protocol review

#### Reviewer
driver-a

#### Verdict
Approved

#### Protocol digest
not-a-digest

#### Digest scope
The frozen task protocol.

#### Outcomes accessed
Yes

#### Review focus
Controls and killing rule.

### Handoff
""",
    )
    errors = _validate_workflow(
        tmp_path,
        monkeypatch,
        text,
        disposition="",
        done=True,
    )
    assert any("Reviewer must differ from the Driver" in error for error in errors)
    assert any("lowercase SHA-256" in error for error in errors)
    assert any("Outcomes accessed as 'No'" in error for error in errors)


def test_third_review_round_requires_recorded_human_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = _workflow_text()
    prefix, accepted_review = accepted.split("### Review", 1)
    revision_review = accepted_review.replace(
        "Accepted",
        "Revision required",
        1,
    )
    three_rounds = (
        prefix
        + "### Review"
        + revision_review
        + "\n### Review"
        + revision_review
        + "\n### Review"
        + accepted_review
    )
    errors = _validate_workflow(
        tmp_path,
        monkeypatch,
        three_rounds,
        disposition="",
        done=True,
    )
    assert any("without prior bounded authorization" in error for error in errors)

    authorization = """
### Revision authorization

#### Authorized by
maintainer-a

#### Additional rounds
1

#### Decision
Authorize one final bounded correction and review.

#### Date
2026-07-29
"""
    authorized = three_rounds.replace(
        "\n### Review" + accepted_review,
        authorization + "\n### Review" + accepted_review,
        1,
    )
    assert (
        _validate_workflow(
            tmp_path,
            monkeypatch,
            authorized,
            disposition="",
            done=True,
        )
        == []
    )

    retroactive = three_rounds + authorization
    errors = _validate_workflow(
        tmp_path,
        monkeypatch,
        retroactive,
        disposition="",
        done=True,
    )
    assert any("without prior bounded authorization" in error for error in errors)


def test_generated_session_brief_is_derived_from_open_dependencies(tmp_path: Path) -> None:
    generator = _load("generate_session_brief")
    tasks = tmp_path / "tasks"
    (tasks / "done").mkdir(parents=True)
    (tasks / "INDEX.md").write_text(
        """# Task index

## Active Tasks

| ID | Title | Status | Depends on |
|----|-------|--------|-----------|
| DOCS-001 | current | in-progress | |
| DOCS-002 | waiting | todo | DOCS-001 |
| DOCS-003 | ready | todo | |
| DOCS-004 | closed | completed | |

## Retired Done Tasks

| ID | Title | Path |
|----|-------|------|
| DOCS-000 | old | `done/DOCS-000-old.md` |
""",
        encoding="utf-8",
    )
    for task_id in ("DOCS-001", "DOCS-002", "DOCS-003", "DOCS-004"):
        (tasks / f"{task_id}-fixture.md").write_text(f"# {task_id} — fixture\n", encoding="utf-8")
    brief = generator.generate(tmp_path)
    assert "`DOCS-001` — current" in brief.split("## Work in progress", 1)[1]
    ready_section = brief.split("## Actionable with no open recorded task dependency", 1)[1].split(
        "## Open tracks", 1
    )[0]
    assert "`DOCS-003` — ready" in ready_section
    assert "`DOCS-002`" not in ready_section
    assert "`DOCS-002` — waiting; open dependencies: `DOCS-001`" in brief
    assert "4 active-table outcomes" in brief
    assert "1 retired tasks" in brief


def _write_report_bundle(
    root: Path,
    *,
    dirty: bool = False,
    status: str = "ok",
    absolute_artifacts: bool = False,
) -> None:
    run = root / "runs" / "current" / "one" / "seed_0"
    run.mkdir(parents=True)
    paths = {
        "target_png": run / "target.png",
        "reconstruction_png": run / "reconstruction.png",
        "error_png": run / "error.png",
        "field_npz": run / "field.npz",
        "history_json": run / "history.json",
        "config_json": run / "config.json",
    }
    for name in ("target_png", "reconstruction_png", "error_png", "field_npz"):
        paths[name].write_bytes(name.encode())
    paths["history_json"].write_text("{}\n", encoding="utf-8")

    repository = {
        "commit": "a" * 40,
        "branch": "docs/fixture",
        "dirty": dirty,
        "status_sha256": ("b" * 64 if dirty else hashlib.sha256(b"").hexdigest()),
    }
    config = {
        "schema": "structsplat.current_pipeline.run.v1",
        "repository": repository,
        "method": "structsplat_best_default",
        "variant": "current",
        "seed": 0,
        "source": {"relative": "one.png", "sha256": "c" * 64},
    }
    paths["config_json"].write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "structsplat.current_pipeline.workflow.v1",
        "command": "python scripts/benchmark.py images report",
        "variants": ["current"],
        "seeds": [0],
        "images": [{"relative": "one.png"}],
        "repository": repository,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    row = {
        "schema": "structsplat.current_pipeline.metric.v1",
        "status": status,
        "error": "" if status == "ok" else "fixture failure",
        "method": "structsplat_best_default",
        "variant": "current",
        "source_id": "one.png",
        "seed": 0,
    }
    if status == "ok":
        artifact_values = {
            name: (str(path.resolve()) if absolute_artifacts else path.relative_to(root).as_posix())
            for name, path in paths.items()
        }
        row.update(
            {
                "psnr": 30.5,
                "ms_ssim": 0.98,
                "total_seconds": 1.25,
                **artifact_values,
                "field_sha256": hashlib.sha256(paths["field_npz"].read_bytes()).hexdigest(),
                "curves": [{"psnr": 30.5}],
                "snapshots": [],
            }
        )
    rows = [row]
    (root / "metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "metrics.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    fields = sorted(key for key in row if key not in {"curves", "snapshots"})
    with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                key: (
                    json.dumps(row.get(key), sort_keys=True, default=str)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key)
                )
                for key in fields
            }
        )
    artifact_links = "".join(
        f'<a href="{path.relative_to(root).as_posix()}">{name}</a>' for name, path in paths.items()
    )
    (root / "index.html").write_text(
        """<!doctype html><title>Fixture 30.5</title>
<a href="manifest.json">manifest</a>
<a href="metrics.json">json</a>
<a href="metrics.jsonl">jsonl</a>
<a href="metrics.csv">csv</a>
"""
        + artifact_links,
        encoding="utf-8",
    )


def test_report_bundle_accepts_only_consistent_clean_portable_result(
    tmp_path: Path,
) -> None:
    checker = _load("check_report_bundle")
    _write_report_bundle(tmp_path)
    assert checker.check_bundle(tmp_path) == []

    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace("30.5", "31.5"), encoding="utf-8"
    )
    (tmp_path / "index.html").write_text(
        (tmp_path / "index.html").read_text(encoding="utf-8")
        + '<a href="../escape.txt">escape</a>',
        encoding="utf-8",
    )
    problems = checker.check_bundle(tmp_path)
    assert any("differs from metrics.json" in problem for problem in problems)
    assert any("escapes or is missing" in problem for problem in problems)


def test_report_bundle_dirty_and_error_overrides_remain_explicit(tmp_path: Path) -> None:
    checker = _load("check_report_bundle")
    dirty_root = tmp_path / "dirty"
    _write_report_bundle(dirty_root, dirty=True)
    assert any("repository was dirty" in problem for problem in checker.check_bundle(dirty_root))
    assert checker.check_bundle(dirty_root, allow_dirty=True) == []

    error_root = tmp_path / "error"
    _write_report_bundle(error_root, status="error")
    assert any(
        "error cell is not claim-ready" in problem for problem in checker.check_bundle(error_root)
    )
    assert checker.check_bundle(error_root, allow_error_cells=True) == []


def test_report_bundle_rejects_incomplete_and_nonfinite_data(tmp_path: Path) -> None:
    checker = _load("check_report_bundle")
    incomplete = tmp_path / "incomplete"
    _write_report_bundle(incomplete)
    (incomplete / "manifest.json").unlink()
    assert "missing required report file: manifest.json" in checker.check_bundle(incomplete)

    nonfinite = tmp_path / "nonfinite"
    _write_report_bundle(nonfinite)
    metrics = (nonfinite / "metrics.json").read_text(encoding="utf-8")
    (nonfinite / "metrics.json").write_text(
        metrics.replace("30.5", "NaN"),
        encoding="utf-8",
    )
    assert any("metrics.json is invalid" in problem for problem in checker.check_bundle(nonfinite))

    absolute = tmp_path / "absolute"
    _write_report_bundle(absolute, absolute_artifacts=True)
    assert any(
        "non-portable absolute path" in problem for problem in checker.check_bundle(absolute)
    )

    escaped = tmp_path / "escaped"
    _write_report_bundle(escaped)
    outside = tmp_path / "outside-metrics.json"
    outside.write_text("[]\n", encoding="utf-8")
    (escaped / "metrics.json").unlink()
    (escaped / "metrics.json").symlink_to(outside)
    assert "required report file escapes the bundle: metrics.json" in checker.check_bundle(escaped)


def test_verify_spine_rejects_duplicate_stages(
    tmp_path: Path,
) -> None:
    checker = _load("check_agent_workflow")
    script = tmp_path / "scripts" / "verify.sh"
    script.parent.mkdir(parents=True)
    commands = "\n".join(checker.EXPECTED_VERIFY_COMMANDS)
    script.write_text(
        "#!/usr/bin/env bash\n" + commands + "\n" + checker.EXPECTED_VERIFY_COMMANDS[0] + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    problems: list[str] = []
    checker.check_verify_script(tmp_path, problems)
    assert any("found 2" in problem for problem in problems)


def test_ci_structural_matrix_covers_the_supported_python_floor(tmp_path: Path) -> None:
    checker = _load("check_agent_workflow")
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    current = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    workflow.write_text(
        current.replace(
            checker.STRUCTURAL_PYTHON_MATRIX,
            'python-version: ["3.11"]',
        ),
        encoding="utf-8",
    )
    problems: list[str] = []
    checker.check_ci(tmp_path, problems)
    assert any("Python 3.10-3.13" in problem for problem in problems)


def _write_claim_fixture(
    root: Path, *, status: str = "supported", proof: str = "`docs/proof.md`"
) -> None:
    (root / "ara" / "logic").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "docs" / "proof.md").write_text("proof\n", encoding="utf-8")
    (root / "ara" / "logic" / "claims.md").write_text(
        f"""# Claims

## C12: Fixture

- **Statement**: Fixture capability.
- **Status**: {status}
- **Provenance**: test
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The fixture fails.
- **Proof**: [{proof}]
- **Dependencies**: []
- **Tags**: fixture
- **From staging**: O01
""",
        encoding="utf-8",
    )


def _claim_findings(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, dict[str, dict[str, str]]]:
    checker = _load("check_ara")
    monkeypatch.setattr(checker, "ROOT", root)
    monkeypatch.setattr(checker, "ARA", root / "ara")
    checker.errors.clear()
    claims = checker.parse_claims()
    checker.check_claim_fields(claims)
    checker.check_claim_statuses(claims)
    checker.check_claim_proofs(claims)
    return checker, claims


def test_ara_proof_paths_cannot_escape_by_traversal_or_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside.md"
    outside.write_text("external\n", encoding="utf-8")
    relative = os.path.relpath(outside, root / "docs")
    _write_claim_fixture(root, proof=f"`docs/{Path(relative).as_posix()}`")
    checker, _ = _claim_findings(root, monkeypatch)
    assert any("escapes repository root" in error for error in checker.errors)

    (root / "docs" / "outside-link.md").symlink_to(outside)
    claims_path = root / "ara" / "logic" / "claims.md"
    claims_path.write_text(
        claims_path.read_text(encoding="utf-8").replace(
            f"`docs/{Path(relative).as_posix()}`",
            "`docs/outside-link.md`",
        ),
        encoding="utf-8",
    )
    checker, _ = _claim_findings(root, monkeypatch)
    assert any("escapes repository root" in error for error in checker.errors)


def test_ara_rejects_unknown_duplicate_and_empty_or_punctuated_status_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _write_claim_fixture(root, status=":;,.", proof="`docs/proof.md`")
    claims_path = root / "ara" / "logic" / "claims.md"
    claims_path.write_text(
        claims_path.read_text(encoding="utf-8").replace(
            "- **From staging**: O01",
            "- **From staging**: O01\n- **Mystery-key** : value\n- **Proof**: [`docs/proof.md`]",
        ),
        encoding="utf-8",
    )
    checker, _ = _claim_findings(root, monkeypatch)
    assert any("has no status disposition" in error for error in checker.errors)
    assert any("unknown field 'Mystery-key'" in error for error in checker.errors)
    assert any("repeats field 'Proof'" in error for error in checker.errors)


def test_punctuated_disposed_status_still_requires_existing_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _write_claim_fixture(root, status="supported:", proof="abcdef0")
    checker, _ = _claim_findings(root, monkeypatch)
    assert any("exists inside the repository" in error for error in checker.errors)
