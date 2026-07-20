from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks import ssp2v_actual_coder as actual
from benchmarks import ssp2v_actual_run as run


class InjectedCrash(RuntimeError):
    """Test-only interruption after an explicitly named durable boundary."""


def _file_snapshot(root: Path) -> dict[str, tuple[int, bytes]]:
    return {
        path.relative_to(root).as_posix(): (
            stat.S_IMODE(os.lstat(path).st_mode),
            path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _write_pending(path: Path, payload: bytes, *, readonly: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = run._pending_publication_path(path, payload)
    pending.write_bytes(payload)
    if readonly:
        pending.chmod(0o444)
    return pending


def test_captured_replay_canonical_read_does_not_enumerate_artifact_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_root = tmp_path / "captured-source"
    captured_root.mkdir()
    (captured_root / "SOURCE_MANIFEST.json").write_bytes(b"{}")
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    artifact = artifact_root / "preflight.json"
    expected = {"schema": "sealed"}
    artifact.write_bytes(run._canonical_json(expected))  # noqa: SLF001

    monkeypatch.setattr(run, "ROOT", captured_root)
    monkeypatch.setenv(  # noqa: SLF001
        run._CAPTURED_REPLAY_SOURCE_ENV, os.fspath(captured_root)
    )
    artifact_root.chmod(0o111)
    try:
        with pytest.raises(PermissionError):
            list(artifact_root.iterdir())
        assert run._read_canonical_object(artifact) == expected  # noqa: SLF001
    finally:
        artifact_root.chmod(0o755)


def test_ordinary_canonical_read_finishes_pending_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(run._CAPTURED_REPLAY_SOURCE_ENV, raising=False)  # noqa: SLF001
    artifact = tmp_path / "preflight.json"
    expected = {"schema": "sealed"}
    payload = run._canonical_json(expected)  # noqa: SLF001
    pending = _write_pending(artifact, payload, readonly=True)

    assert run._read_canonical_object(artifact) == expected  # noqa: SLF001
    assert artifact.read_bytes() == payload
    assert not pending.exists()


@pytest.mark.parametrize("hostility", ["missing-manifest", "mismatched-root"])
def test_captured_replay_canonical_read_rejects_invalid_source_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hostility: str
) -> None:
    captured_root = tmp_path / "captured-source"
    captured_root.mkdir()
    artifact = captured_root / "preflight.json"
    artifact.write_bytes(run._canonical_json({"schema": "sealed"}))  # noqa: SLF001
    marker = captured_root
    if hostility == "mismatched-root":
        (captured_root / "SOURCE_MANIFEST.json").write_bytes(b"{}")
        marker = tmp_path / "foreign-source"
        marker.mkdir()
        (marker / "SOURCE_MANIFEST.json").write_bytes(b"{}")

    monkeypatch.setattr(run, "ROOT", captured_root)
    monkeypatch.setenv(run._CAPTURED_REPLAY_SOURCE_ENV, os.fspath(marker))  # noqa: SLF001
    with pytest.raises(run.LifecycleError, match="source-root marker"):
        run._read_canonical_object(artifact)  # noqa: SLF001


@pytest.mark.parametrize("hostility", ["symlink", "noncanonical"])
def test_captured_replay_canonical_read_keeps_named_file_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hostility: str
) -> None:
    captured_root = tmp_path / "captured-source"
    captured_root.mkdir()
    (captured_root / "SOURCE_MANIFEST.json").write_bytes(b"{}")
    artifact = captured_root / "preflight.json"
    if hostility == "symlink":
        target = tmp_path / "foreign.json"
        target.write_bytes(run._canonical_json({"schema": "sealed"}))  # noqa: SLF001
        artifact.symlink_to(target)
    else:
        artifact.write_bytes(b'{"schema": "sealed"}\n')

    monkeypatch.setattr(run, "ROOT", captured_root)
    monkeypatch.setenv(  # noqa: SLF001
        run._CAPTURED_REPLAY_SOURCE_ENV, os.fspath(captured_root)
    )
    match = "symlink" if hostility == "symlink" else "not canonical"
    with pytest.raises(run.LifecycleError, match=match):
        run._read_canonical_object(artifact)  # noqa: SLF001


@pytest.mark.parametrize(
    "checkpoint",
    ["writable-partial", "readonly-before-link", "readonly-after-link"],
)
def test_immutable_publication_recovers_every_durable_checkpoint(
    tmp_path: Path, checkpoint: str
) -> None:
    path = tmp_path / "artifact.bin"
    payload = b"complete immutable publication\n"
    pending = run._pending_publication_path(path, payload)
    if checkpoint == "writable-partial":
        pending.write_bytes(payload[:9])
    else:
        pending.write_bytes(payload)
        pending.chmod(0o444)
        if checkpoint == "readonly-after-link":
            os.link(pending, path)

    run._exclusive_write(path, payload)

    assert path.read_bytes() == payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert path.stat().st_nlink == 1
    assert not pending.exists()


@pytest.mark.parametrize(
    "name",
    [
        ".pending-orphan",
        ".pending-artifact.bin-not-a-sha256",
        ".pending-.pending-nested-" + "0" * 64,
    ],
)
def test_pending_namespace_rejects_every_malformed_entry(
    tmp_path: Path, name: str
) -> None:
    malformed = tmp_path / name
    malformed.write_bytes(b"hostile")

    with pytest.raises(run.LifecycleError, match="malformed pending"):
        run._recover_pending_directory(tmp_path)

    assert malformed.read_bytes() == b"hostile"


@pytest.mark.parametrize(
    "hostility", ["symlink", "hardlink", "digest-drift", "final-conflict"]
)
def test_pending_publication_rejects_links_drift_and_conflict(
    tmp_path: Path, hostility: str
) -> None:
    target = tmp_path / "artifact.bin"
    payload = b"authorized bytes"
    pending = run._pending_publication_path(target, payload)
    if hostility in {"symlink", "hardlink"}:
        outside = tmp_path / "outside.bin"
        outside.write_bytes(payload)
        outside.chmod(0o444)
        if hostility == "symlink":
            pending.symlink_to(outside)
        else:
            os.link(outside, pending)
    else:
        pending.write_bytes(b"drift" if hostility == "digest-drift" else payload)
        pending.chmod(0o444)
        if hostility == "final-conflict":
            target.write_bytes(b"different final bytes")
            target.chmod(0o444)

    expected = {
        "symlink": "regular no-link",
        "hardlink": "external hard link",
        "digest-drift": "digest drift",
        "final-conflict": "conflicts with final",
    }[hostility]
    with pytest.raises(run.LifecycleError, match=expected):
        run._recover_pending_directory(tmp_path)


def test_journal_recovers_exact_pending_terminal_event_without_a_suffix(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal/events"
    journal = run.EventJournal(root)
    first = journal.append(
        stage="preflight",
        operation="open-read-complete",
        path_class="repository_source",
        purpose="stable prefix",
        path="repository:first",
    )
    terminal = journal.append(
        stage="preflight",
        operation="open-read-complete",
        path_class="repository_source",
        purpose="terminal event interrupted after fsync",
        path="repository:terminal",
    )
    terminal_path = root / "00000001.json"
    payload = terminal_path.read_bytes()
    pending = run._pending_publication_path(terminal_path, payload)
    terminal_path.rename(pending)

    events = run.EventJournal.verify(root)

    assert events == [first, terminal]
    assert terminal_path.read_bytes() == payload
    assert not pending.exists()
    assert sorted(path.name for path in root.iterdir()) == [
        "00000000.json",
        "00000001.json",
    ]
    stable = _file_snapshot(root)
    assert run.EventJournal.verify(root) == events
    assert _file_snapshot(root) == stable


def _seed_completed_predecessor(
    outdir: Path,
) -> tuple[run.EventJournal, str, dict[str, Any]]:
    journal = run.EventJournal(outdir / "journal/events")
    payload = b"synthetic sealed preflight\n"
    authorization = journal.append(
        stage="preflight",
        operation="authorize-create-immutable",
        path_class="artifact_runtime",
        purpose="authorize synthetic predecessor",
        path="artifact:preflight.json",
    )
    binding = {"length": journal.length, "head_sha256": authorization["event_sha256"]}
    journal.append(
        stage="preflight",
        operation="create-immutable-complete",
        path_class="artifact_runtime",
        purpose="complete synthetic predecessor",
        path="artifact:preflight.json",
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        artifact_bytes=len(payload),
    )
    predecessor_seal = "a" * 64
    journal.bind_stage_seal("preflight", predecessor_seal)
    return journal, predecessor_seal, binding


def _authorized_stage_record(
    outdir: Path,
) -> tuple[run.EventJournal, dict[str, Any], dict[str, Any]]:
    journal, predecessor_seal, predecessor_binding = _seed_completed_predecessor(outdir)
    authorization = journal.append(
        stage="import-streams",
        operation="authorize-create-immutable",
        path_class="artifact_runtime",
        purpose="authorize interrupted import manifest",
        path="artifact:import_streams.json",
        prior_stage_seal=predecessor_seal,
    )
    record = actual.seal_record(
        {
            "schema": run.IMPORT_STREAMS_SCHEMA,
            "protocol": run.PROTOCOL,
            "task_sha256": run.TASK_SHA256,
            "stage": "import-streams",
            "prior_stage": "preflight",
            "prior_stage_seal": predecessor_seal,
            "state": "synthetic-target-free-recovery-fixture",
            "journal": {
                "length": journal.length,
                "head_sha256": authorization["event_sha256"],
            },
            "confirmation_payloads_accessed": False,
        },
        "import_streams_sha256",
    )
    return journal, record, predecessor_binding


def test_stage_manifest_recovers_readonly_pending_then_exact_completion(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "artifact"
    journal, record, predecessor_binding = _authorized_stage_record(outdir)
    path = outdir / "import_streams.json"
    payload = run._canonical_json(record)
    pending = _write_pending(path, payload, readonly=True)

    run._recover_interrupted_stage_completion(outdir, "import-streams")

    events = run.EventJournal.verify(journal.root)
    binding = record["journal"]
    run._verify_stage_journal_segment(
        events,
        binding,
        stage="import-streams",
        manifest_path="import_streams.json",
        manifest_payload=payload,
        prior_stage_seal=str(record["prior_stage_seal"]),
        predecessor_binding=predecessor_binding,
    )
    assert path.read_bytes() == payload
    assert not pending.exists()
    assert len(events) == int(binding["length"]) + 1
    assert events[-1]["operation"] == "create-immutable-complete"
    assert events[-1]["artifact_sha256"] == hashlib.sha256(payload).hexdigest()
    assert events[-1]["artifact_bytes"] == len(payload)

    stable = _file_snapshot(outdir)
    run._recover_interrupted_stage_completion(outdir, "import-streams")
    assert _file_snapshot(outdir) == stable


def test_stage_resume_after_authorization_only_preserves_prefix_and_finishes(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "artifact"
    journal, predecessor_seal, predecessor_binding = _seed_completed_predecessor(outdir)
    orphan = journal.append(
        stage="import-streams",
        operation="authorize-create-immutable",
        path_class="artifact_runtime",
        purpose="authorization durable before interruption",
        path="artifact:import_streams.json",
        prior_stage_seal=predecessor_seal,
    )
    prefix = _file_snapshot(journal.root)

    record = run._finalize_stage(
        outdir,
        journal,
        stage="import-streams",
        schema=run.IMPORT_STREAMS_SCHEMA,
        prior_stage_seal=predecessor_seal,
        body={"state": "synthetic-target-free-recovery-fixture"},
    )

    assert all(_file_snapshot(journal.root)[name] == value for name, value in prefix.items())
    events = run.EventJournal.verify(journal.root)
    payload = run._canonical_json(record)
    run._verify_stage_journal_segment(
        events,
        record["journal"],
        stage="import-streams",
        manifest_path="import_streams.json",
        manifest_payload=payload,
        prior_stage_seal=predecessor_seal,
        predecessor_binding=predecessor_binding,
    )
    authorizations = [
        event
        for event in events
        if event["stage"] == "import-streams"
        and event["operation"] == "authorize-create-immutable"
    ]
    assert authorizations[0]["event_sha256"] == orphan["event_sha256"]
    assert authorizations[-1]["event_sha256"] == record["journal"]["head_sha256"]
    assert events[-1]["operation"] == "create-immutable-complete"


def test_stage_recovery_never_appends_completion_after_a_journal_suffix(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "artifact"
    journal, record, predecessor_binding = _authorized_stage_record(outdir)
    payload = run._canonical_json(record)
    run._exclusive_write(outdir / "import_streams.json", payload)
    journal.append(
        stage="import-streams",
        operation="open-read-complete",
        path_class="artifact_stream",
        purpose="hostile suffix after final authorization",
        path="artifact:streams/foreign.sspl1",
        prior_stage_seal=str(record["prior_stage_seal"]),
    )
    before = _file_snapshot(outdir)

    run._recover_interrupted_stage_completion(outdir, "import-streams")

    assert _file_snapshot(outdir) == before
    events = run.EventJournal.verify(journal.root)
    with pytest.raises(run.LifecycleError, match="completion"):
        run._verify_stage_journal_segment(
            events,
            record["journal"],
            stage="import-streams",
            manifest_path="import_streams.json",
            manifest_payload=payload,
            prior_stage_seal=str(record["prior_stage_seal"]),
            predecessor_binding=predecessor_binding,
        )


def _candidate_result(label: str, *, terminal: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        record={
            "status": "terminal_lloyd_failure" if terminal else "valid_complete_candidate_set",
            "fixture": label,
        },
        baseline_blobs={
            "sspl1": f"{label}-sspl1".encode(),
            "ssp2z": f"{label}-ssp2z".encode(),
        },
        variant_blobs={
            "flat_s2_k64_l192": f"{label}-flat".encode(),
            "rvq_s2_k64": f"{label}-rvq".encode(),
        },
    )


def test_candidate_blob_resume_completes_interrupted_prefix_byte_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "artifact"
    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    result = _candidate_result("cell-0")
    original_write = run._exclusive_write
    writes = 0

    def crash_after_second_link(
        path: Path, payload: bytes, *, readonly: bool = True
    ) -> None:
        nonlocal writes
        original_write(path, payload, readonly=readonly)
        writes += 1
        if writes == 2:
            raise InjectedCrash("candidate blob linked before interruption")

    monkeypatch.setattr(run, "_exclusive_write", crash_after_second_link)
    with pytest.raises(InjectedCrash):
        run._prepare_candidate_blob_directory(
            outdir, image_id=image_id, bits=bits, result=result
        )
    monkeypatch.setattr(run, "_exclusive_write", original_write)

    final, files = run._prepare_candidate_blob_directory(
        outdir, image_id=image_id, bits=bits, result=result
    )
    assert {
        path.relative_to(final).as_posix(): path.read_bytes()
        for path in final.rglob("*")
        if path.is_file()
    } == files
    stable = _file_snapshot(outdir)
    second_final, second_files = run._prepare_candidate_blob_directory(
        outdir, image_id=image_id, bits=bits, result=result
    )
    assert second_final == final
    assert second_files == files
    assert _file_snapshot(outdir) == stable


def _candidate_journal(outdir: Path, prior_stage_seal: str) -> run.EventJournal:
    journal = run.EventJournal(outdir / "journal/events")
    journal.bind_stage_seal("import-streams", prior_stage_seal)
    return journal


def test_candidate_cell_resume_recovers_missing_commit_once_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir = tmp_path / "artifact"
    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    prior_stage_seal = "b" * 64
    result = _candidate_result("cell-commit")
    journal = _candidate_journal(outdir, prior_stage_seal)
    monkeypatch.setattr(run, "_verify_candidate_cell", lambda _root, value: dict(value))
    original_append = journal.append

    def crash_before_commit(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("operation") == "commit-immutable-cell-directory":
            raise InjectedCrash("cell.json durable before journal commit")
        return original_append(**kwargs)

    monkeypatch.setattr(journal, "append", crash_before_commit)
    with pytest.raises(InjectedCrash):
        run._commit_candidate_cell(
            outdir,
            journal,
            image_id=image_id,
            bits=bits,
            execution_result=result,
            cold_decodes={},
            diagnostics={"target_accessed": False},
            prior_stage_seal=prior_stage_seal,
        )
    cell_path = outdir / run._candidate_cell_relative(image_id, bits) / "cell.json"
    assert cell_path.exists()
    assert not [
        event
        for event in run.EventJournal.verify(journal.root)
        if event["operation"] == "commit-immutable-cell-directory"
    ]

    monkeypatch.setattr(journal, "append", original_append)
    first = run._commit_candidate_cell(
        outdir,
        journal,
        image_id=image_id,
        bits=bits,
        execution_result=result,
        cold_decodes={},
        diagnostics={"target_accessed": False},
        prior_stage_seal=prior_stage_seal,
    )
    commits = [
        event
        for event in run.EventJournal.verify(journal.root)
        if event["operation"] == "commit-immutable-cell-directory"
    ]
    assert len(commits) == 1
    assert commits[0]["artifact_sha256"] == hashlib.sha256(cell_path.read_bytes()).hexdigest()
    assert commits[0]["artifact_bytes"] == cell_path.stat().st_size
    stable = _file_snapshot(outdir)

    second = run._commit_candidate_cell(
        outdir,
        journal,
        image_id=image_id,
        bits=bits,
        execution_result=result,
        cold_decodes={},
        diagnostics={"target_accessed": False},
        prior_stage_seal=prior_stage_seal,
    )
    assert second == first
    assert _file_snapshot(outdir) == stable


def test_candidate_cell_resume_rejects_conflicting_commit_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir = tmp_path / "artifact"
    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    prior_stage_seal = "c" * 64
    result = _candidate_result("receipt-conflict")
    journal = _candidate_journal(outdir, prior_stage_seal)
    monkeypatch.setattr(run, "_verify_candidate_cell", lambda _root, value: dict(value))
    original_append = journal.append
    monkeypatch.setattr(
        journal,
        "append",
        lambda **kwargs: (
            (_ for _ in ()).throw(InjectedCrash("before correct commit"))
            if kwargs.get("operation") == "commit-immutable-cell-directory"
            else original_append(**kwargs)
        ),
    )
    with pytest.raises(InjectedCrash):
        run._commit_candidate_cell(
            outdir,
            journal,
            image_id=image_id,
            bits=bits,
            execution_result=result,
            cold_decodes={},
            diagnostics={},
            prior_stage_seal=prior_stage_seal,
        )
    monkeypatch.setattr(journal, "append", original_append)
    cell_directory = outdir / run._candidate_cell_relative(image_id, bits)
    journal.append(
        stage="fit-encode-cold-decode-dev",
        operation="commit-immutable-cell-directory",
        path_class="artifact_stream",
        purpose="hostile conflicting candidate receipt",
        path=run._artifact_label(outdir, cell_directory),
        prior_stage_seal=prior_stage_seal,
        artifact_sha256="0" * 64,
        artifact_bytes=1,
    )

    with pytest.raises(run.LifecycleError, match="commit.*receipt"):
        run._commit_candidate_cell(
            outdir,
            journal,
            image_id=image_id,
            bits=bits,
            execution_result=result,
            cold_decodes={},
            diagnostics={},
            prior_stage_seal=prior_stage_seal,
        )


def test_terminal_candidate_commit_forbids_any_later_cell_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir = tmp_path / "artifact"
    prior_stage_seal = "d" * 64
    journal = _candidate_journal(outdir, prior_stage_seal)
    monkeypatch.setattr(run, "_verify_candidate_cell", lambda _root, value: dict(value))
    first_image = actual.FROZEN_IMAGES[0]
    first_bits = actual.FROZEN_TUPLES[0]
    run._commit_candidate_cell(
        outdir,
        journal,
        image_id=first_image,
        bits=first_bits,
        execution_result=_candidate_result("terminal", terminal=True),
        cold_decodes={},
        diagnostics={},
        prior_stage_seal=prior_stage_seal,
    )
    before = _file_snapshot(outdir)

    with pytest.raises(run.LifecycleError, match="terminal Lloyd"):
        run._commit_candidate_cell(
            outdir,
            journal,
            image_id=first_image,
            bits=actual.FROZEN_TUPLES[1],
            execution_result=_candidate_result("illegal-suffix"),
            cold_decodes={},
            diagnostics={},
            prior_stage_seal=prior_stage_seal,
        )

    assert _file_snapshot(outdir) == before


@pytest.mark.parametrize(
    ("hostility", "receipt_order", "expected"),
    [
        ("missing", (0,), "candidate commit journal differs"),
        ("extra", (0, 1, 2), "candidate commit journal differs"),
        ("reordered", (1, 0), "candidate commit journal differs"),
        ("mismatched", (0, 1), "candidate cell commit receipt is invalid"),
    ],
)
def test_candidate_stage_rejects_hostile_cell_commit_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hostility: str,
    receipt_order: tuple[int, ...],
    expected: str,
) -> None:
    outdir = tmp_path / "artifact"
    import_seal = "e" * 64
    journal = _candidate_journal(outdir, import_seal)
    cells: list[dict[str, Any]] = []
    payloads: list[bytes] = []
    labels: list[str] = []
    for index, (image_id, bits) in enumerate(run._expected_identities()[:3]):
        relative = run._candidate_cell_relative(image_id, bits)
        cell = {
            "image_id": image_id,
            "bit_tuple": list(bits),
            "state": "terminal_lloyd_failure" if index == 1 else "complete_candidate_cell",
            "relative_path": relative.as_posix(),
            "cold_decodes": {},
        }
        payload = run._canonical_json(cell)
        run._exclusive_write(outdir / relative / "cell.json", payload)
        cells.append(cell)
        payloads.append(payload)
        labels.append(run._artifact_label(outdir, outdir / relative))

    for position, index in enumerate(receipt_order):
        payload = payloads[index]
        journal.append(
            stage="fit-encode-cold-decode-dev",
            operation="commit-immutable-cell-directory",
            path_class="artifact_stream",
            purpose="commit target-free fit/encode/cold-decode cell",
            path=labels[index],
            prior_stage_seal=import_seal,
            artifact_sha256=hashlib.sha256(payload).hexdigest(),
            artifact_bytes=(
                len(payload) + 1
                if hostility == "mismatched" and position == 0
                else len(payload)
            ),
        )
    stage_record = {
        "prior_stage_seal": import_seal,
        "state": "valid_terminal_lloyd_failure",
        "cells": cells[:2],
        "cell_count": 2,
        "candidate_set_sha256": None,
        "target_import_authorized": False,
        "target_or_pixel_payloads_opened": False,
    }
    monkeypatch.setattr(
        run,
        "verify_import_streams",
        lambda _root, **_kwargs: {},
    )
    monkeypatch.setattr(
        run,
        "_verify_stage_base",
        lambda _root, **_kwargs: stage_record,
    )
    monkeypatch.setattr(run, "_verify_candidate_cell", lambda _root, value: dict(value))

    with pytest.raises(run.LifecycleError, match=expected):
        run.verify_candidate_stage(outdir)


def test_replay_stage_base_rejects_any_journal_suffix_after_completion(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "artifact"
    journal = run.EventJournal(outdir / "journal/events")
    benchmark_seal = "f" * 64
    analysis_seal = "1" * 64
    journal.bind_stage_seal("benchmark-dev", benchmark_seal)
    analysis_authorization = journal.append(
        stage="analyze",
        operation="authorize-create-immutable",
        path_class="artifact_runtime",
        purpose="authorize synthetic analysis predecessor",
        path="artifact:analysis_record.json",
        prior_stage_seal=benchmark_seal,
    )
    analysis_binding = {
        "length": journal.length,
        "head_sha256": analysis_authorization["event_sha256"],
    }
    analysis_payload = b"synthetic completed analysis\n"
    journal.append(
        stage="analyze",
        operation="create-immutable-complete",
        path_class="artifact_runtime",
        purpose="complete synthetic analysis predecessor",
        path="artifact:analysis_record.json",
        prior_stage_seal=benchmark_seal,
        artifact_sha256=hashlib.sha256(analysis_payload).hexdigest(),
        artifact_bytes=len(analysis_payload),
    )
    analysis_record = {
        "analysis_record_sha256": analysis_seal,
        "journal": analysis_binding,
    }
    journal.bind_stage_seal("analyze", analysis_seal)
    replay_authorization = journal.append(
        stage="replay",
        operation="authorize-create-immutable",
        path_class="artifact_runtime",
        purpose="authorize synthetic terminal replay manifest",
        path="artifact:replay.json",
        prior_stage_seal=analysis_seal,
    )
    replay_record = actual.seal_record(
        {
            "schema": run.REPLAY_SCHEMA,
            "protocol": run.PROTOCOL,
            "task_sha256": run.TASK_SHA256,
            "stage": "replay",
            "prior_stage": "analyze",
            "prior_stage_seal": analysis_seal,
            "journal": {
                "length": journal.length,
                "head_sha256": replay_authorization["event_sha256"],
            },
            "confirmation_payloads_accessed": False,
        },
        "replay_sha256",
    )
    replay_payload = run._canonical_json(replay_record)
    run._exclusive_write(outdir / "replay.json", replay_payload)
    journal.append(
        stage="replay",
        operation="create-immutable-complete",
        path_class="artifact_runtime",
        purpose="complete synthetic terminal replay manifest",
        path="artifact:replay.json",
        prior_stage_seal=analysis_seal,
        artifact_sha256=hashlib.sha256(replay_payload).hexdigest(),
        artifact_bytes=len(replay_payload),
    )
    journal.append(
        stage="replay",
        operation="open-read-complete",
        path_class="artifact_runtime",
        purpose="forbidden suffix after terminal replay completion",
        path="artifact:replay-suffix.json",
        prior_stage_seal=analysis_seal,
    )

    with pytest.raises(run.LifecycleError, match="replay terminal manifest.*suffix"):
        run._verify_stage_base(
            outdir,
            stage="replay",
            schema=run.REPLAY_SCHEMA,
            prior_record=analysis_record,
        )
