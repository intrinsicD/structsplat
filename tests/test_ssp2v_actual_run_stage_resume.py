from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from benchmarks import ssp2v_actual_run as run


class UnexpectedWork(AssertionError):
    """A supposedly frozen stage prefix attempted scientific re-execution."""


class _NoOpGuard:
    def register(self, _path: Path, _path_class: str) -> None:
        return None

    def read_bytes(self, _path: Path, **_authorization: Any) -> bytes:
        return b""


def _candidate_result(label: str, *, terminal: bool) -> SimpleNamespace:
    status = "terminal_lloyd_failure" if terminal else "valid_complete_candidate_set"
    return SimpleNamespace(
        record={"status": status, "fixture": label},
        baseline_blobs={"sspl1": f"{label}-sspl1".encode()},
        variant_blobs={"flat_s2_k64_l192": f"{label}-variant".encode()},
    )


@pytest.mark.parametrize("authorization_only", [False, True])
def test_terminal_lloyd_second_cell_resumes_exact_prefix_without_reexecution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorization_only: bool,
) -> None:
    outdir = tmp_path / "artifact"
    prior_stage_seal = "a" * 64
    identities = list(run._expected_identities()[:2])
    journal = run.EventJournal(outdir / "journal/events")
    journal.bind_stage_seal("import-streams", prior_stage_seal)
    monkeypatch.setattr(run, "_verify_candidate_cell", lambda _root, value: dict(value))

    committed = []
    for index, (image_id, bits) in enumerate(identities):
        committed.append(
            run._commit_candidate_cell(
                outdir,
                journal,
                image_id=image_id,
                bits=bits,
                execution_result=_candidate_result(f"cell-{index}", terminal=index == 1),
                cold_decodes={},
                diagnostics={"decision_use": False},
                prior_stage_seal=prior_stage_seal,
            )
        )
    frozen_cells = {
        outdir / run._candidate_cell_relative(image_id, bits) / "cell.json": (
            outdir / run._candidate_cell_relative(image_id, bits) / "cell.json"
        ).read_bytes()
        for image_id, bits in identities
    }
    if authorization_only:
        journal.append(
            stage="fit-encode-cold-decode-dev",
            operation="authorize-create-immutable",
            path_class="artifact_runtime",
            purpose="authorize final fit-encode-cold-decode-dev manifest",
            path="artifact:candidate_stage.json",
            prior_stage_seal=prior_stage_seal,
        )

    preflight = {
        "native": {"lloyd": {"path": "lloyd.so", "bytes": 1, "sha256": "b" * 64}},
        "operational_upstream_roots": {"comp008": "/synthetic/comp008"},
    }
    authority = {
        "comp008": {"targets": [{"relative_path": "targets/unused.png"}]},
        "comp009": {
            "cells": [],
            "captured_source_archive": {"relative_path": "runtime/source.tar"},
            "captured_source_manifest": {},
            "source_manifest_sha256": "c" * 64,
        },
    }
    monkeypatch.setattr(run, "_expected_identities", lambda: list(identities))
    monkeypatch.setattr(run, "_recover_interrupted_stage_completion", lambda *_args: None)
    monkeypatch.setattr(
        run,
        "_stage_guard",
        lambda _root, _stage: (journal, _NoOpGuard(), {"preflight": preflight}),
    )
    monkeypatch.setattr(
        run,
        "verify_import_streams",
        lambda _root: {"import_streams_sha256": prior_stage_seal, "streams": []},
    )
    monkeypatch.setattr(run, "_authority_copy", lambda _root, _record: authority)
    monkeypatch.setattr(
        run,
        "_native_arithmetic_record",
        lambda _root, _record: (
            outdir / "runtime/native/arithmetic.so",
            {"path": "arithmetic.so", "bytes": 1, "sha256": "d" * 64},
        ),
    )
    monkeypatch.setattr(run, "_verify_regular_record", lambda *_args: None)
    monkeypatch.setattr(run, "safe_extract_source_archive", lambda *_args, **_kwargs: None)

    production_calls: list[str] = []

    def unexpected_production(*_args: Any, **_kwargs: Any) -> Any:
        production_calls.append("produce")
        raise UnexpectedWork("terminal prefix retried candidate production")

    monkeypatch.setattr(run, "_run_producing_cell_worker", unexpected_production)
    monkeypatch.setattr(run, "_run_cold_decode_sample", unexpected_production)

    resumed = run.fit_encode_cold_decode(outdir)

    assert resumed["state"] == "valid_terminal_lloyd_failure"
    assert resumed["cells"] == committed
    assert resumed["cell_count"] == 2
    assert resumed["target_import_authorized"] is False
    assert production_calls == []
    assert {path: path.read_bytes() for path in frozen_cells} == frozen_cells


def _append_cell_receipt(
    *,
    outdir: Path,
    journal: run.EventJournal,
    stage: str,
    prior_stage_seal: str,
    destination: Path,
    payload: bytes,
    purpose: str,
    artifact_sha256: str | None = None,
    artifact_bytes: int | None = None,
) -> dict[str, Any]:
    run._exclusive_write(destination, payload)
    return journal.append(
        stage=stage,
        operation="create-immutable-complete",
        path_class="artifact_runtime",
        purpose=purpose,
        path=run._artifact_label(outdir, destination),
        prior_stage_seal=prior_stage_seal,
        artifact_sha256=(
            hashlib.sha256(payload).hexdigest() if artifact_sha256 is None else artifact_sha256
        ),
        artifact_bytes=len(payload) if artifact_bytes is None else artifact_bytes,
    )


def _quality_candidate(image_id: str, bits: Sequence[int]) -> dict[str, Any]:
    exact_sha = "1" * 64
    variant_sha = "2" * 64
    artifacts = [
        {"path": "exact/sspl1.bin", "bytes": 1, "sha256": exact_sha},
        *[
            {
                "path": f"variants/{label}.ssp2v",
                "bytes": 1,
                "sha256": variant_sha,
            }
            for label in run.execute.VARIANT_LABELS
        ],
    ]
    return {
        "image_id": image_id,
        "bit_tuple": list(bits),
        "cell_record_sha256": "3" * 64,
        "relative_path": run._candidate_cell_relative(image_id, bits).as_posix(),
        "artifacts": artifacts,
        "execution": {
            "selected_q": {"name": "sspl1"},
            "exact_formats": {
                "sspl1": {
                    "complete_bytes": 1,
                    "blob_sha256": exact_sha,
                    "symbol_sha256": "4" * 64,
                    "boundary": {"state_sha256": "5" * 64},
                }
            },
            "variants": {
                label: {
                    "stream": {
                        "complete_bytes": 1,
                        "blob_sha256": variant_sha,
                        "symbol_sha256": "6" * 64,
                        "boundary": {"state_sha256": "7" * 64},
                    }
                }
                for label in run.execute.VARIANT_LABELS
            },
        },
    }


def _stored_quality_cell(image_id: str, bits: Sequence[int]) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "bit_tuple": list(bits),
        "selection": {"state": "method_failure"},
    }


def _patch_quality_stage(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outdir: Path,
    journal: run.EventJournal,
    identities: Sequence[tuple[str, tuple[int, int, int, int]]],
    target_copy_sha256: str,
) -> list[str]:
    candidates = [_quality_candidate(image_id, bits) for image_id, bits in identities]
    targets = [
        {
            "image_id": image_id,
            "rgb_copy": {"relative_path": f"targets/{image_id}.rgb"},
        }
        for image_id in dict.fromkeys(image_id for image_id, _bits in identities)
    ]
    preflight = {
        "renderer": {},
        "renderer_proof": {"same_worker_metric_repeat": {"lpips_state_sha256": "8" * 64}},
    }
    monkeypatch.setattr(run, "_expected_identities", lambda: list(identities))
    monkeypatch.setattr(run, "_recover_interrupted_stage_completion", lambda *_args: None)
    monkeypatch.setattr(
        run,
        "_stage_guard",
        lambda _root, _stage: (journal, _NoOpGuard(), {"preflight": preflight}),
    )
    monkeypatch.setattr(
        run,
        "verify_candidate_stage",
        lambda _root: {"candidate_set_sha256": "9" * 64, "cells": candidates},
    )
    monkeypatch.setattr(
        run,
        "verify_target_copy",
        lambda _root: {
            "state": "complete_target_copy",
            "target_copy_sha256": target_copy_sha256,
            "targets": targets,
        },
    )
    monkeypatch.setattr(
        run,
        "_native_arithmetic_record",
        lambda _root, _record: (
            outdir / "runtime/native/arithmetic.so",
            {"path": "arithmetic.so", "bytes": 1, "sha256": "a" * 64},
        ),
    )
    monkeypatch.setattr(
        run, "_verify_quality_cell", lambda _root, value, *_args, **_kwargs: dict(value)
    )
    render_calls: list[str] = []

    def unexpected_render(*_args: Any, **_kwargs: Any) -> Any:
        render_calls.append("render")
        raise UnexpectedWork("committed quality cell was rendered again")

    monkeypatch.setattr(run, "_run_render_metric_task", unexpected_render)
    return render_calls


def test_journaled_immutable_receipt_binds_payload_hash_and_size(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "artifact"
    prior_stage_seal = "b" * 64
    journal = run.EventJournal(outdir / "journal/events")
    journal.bind_stage_seal("import-targets", prior_stage_seal)
    payload = b"synthetic quality cell\n"
    destination = outdir / run._quality_cell_relative(*run._expected_identities()[0])

    run._journaled_immutable(
        journal,
        outdir,
        destination,
        payload,
        stage="quality-select-dev",
        path_class="artifact_runtime",
        purpose="persist complete nine-arm quality cell",
        prior_stage_seal=prior_stage_seal,
    )

    receipt = run.EventJournal.verify(journal.root)[-1]
    assert receipt.get("artifact_sha256") == hashlib.sha256(payload).hexdigest()
    assert receipt.get("artifact_bytes") == len(payload)


def test_deleted_committed_quality_cell_fails_before_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir = tmp_path / "artifact"
    prior_stage_seal = "c" * 64
    identity = run._expected_identities()[0]
    journal = run.EventJournal(outdir / "journal/events")
    journal.bind_stage_seal("import-targets", prior_stage_seal)
    destination = outdir / run._quality_cell_relative(*identity)
    payload = run._canonical_json(_stored_quality_cell(*identity))
    _append_cell_receipt(
        outdir=outdir,
        journal=journal,
        stage="quality-select-dev",
        prior_stage_seal=prior_stage_seal,
        destination=destination,
        payload=payload,
        purpose="persist complete nine-arm quality cell",
    )
    destination.unlink()
    render_calls = _patch_quality_stage(
        monkeypatch,
        outdir=outdir,
        journal=journal,
        identities=[identity],
        target_copy_sha256=prior_stage_seal,
    )

    with pytest.raises(run.LifecycleError, match="receipt|committed|missing|prefix|deleted"):
        run.quality_select(outdir)

    assert render_calls == []


@pytest.mark.parametrize("hostility", ["hash", "size"])
def test_quality_resume_rejects_corrupt_cell_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hostility: str,
) -> None:
    outdir = tmp_path / "artifact"
    prior_stage_seal = "d" * 64
    identity = run._expected_identities()[0]
    journal = run.EventJournal(outdir / "journal/events")
    journal.bind_stage_seal("import-targets", prior_stage_seal)
    destination = outdir / run._quality_cell_relative(*identity)
    payload = run._canonical_json(_stored_quality_cell(*identity))
    _append_cell_receipt(
        outdir=outdir,
        journal=journal,
        stage="quality-select-dev",
        prior_stage_seal=prior_stage_seal,
        destination=destination,
        payload=payload,
        purpose="persist complete nine-arm quality cell",
        artifact_sha256="0" * 64 if hostility == "hash" else None,
        artifact_bytes=len(payload) + 1 if hostility == "size" else None,
    )
    render_calls = _patch_quality_stage(
        monkeypatch,
        outdir=outdir,
        journal=journal,
        identities=[identity],
        target_copy_sha256=prior_stage_seal,
    )

    with pytest.raises(run.LifecycleError, match="receipt|hash|bytes|size"):
        run.quality_select(outdir)

    assert render_calls == []


def test_quality_resume_rejects_nonprefix_cell_receipt_before_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir = tmp_path / "artifact"
    prior_stage_seal = "e" * 64
    identities = list(run._expected_identities()[:2])
    journal = run.EventJournal(outdir / "journal/events")
    journal.bind_stage_seal("import-targets", prior_stage_seal)
    second = identities[1]
    destination = outdir / run._quality_cell_relative(*second)
    payload = run._canonical_json(_stored_quality_cell(*second))
    _append_cell_receipt(
        outdir=outdir,
        journal=journal,
        stage="quality-select-dev",
        prior_stage_seal=prior_stage_seal,
        destination=destination,
        payload=payload,
        purpose="persist complete nine-arm quality cell",
    )
    render_calls = _patch_quality_stage(
        monkeypatch,
        outdir=outdir,
        journal=journal,
        identities=identities,
        target_copy_sha256=prior_stage_seal,
    )

    with pytest.raises(run.LifecycleError, match="receipt|order|prefix"):
        run.quality_select(outdir)

    assert render_calls == []


def test_deleted_committed_benchmark_cell_fails_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir = tmp_path / "artifact"
    prior_stage_seal = "f" * 64
    image_id, bits = run._expected_identities()[0]
    journal = run.EventJournal(outdir / "journal/events")
    journal.bind_stage_seal("quality-select-dev", prior_stage_seal)
    destination = outdir / run._benchmark_cell_relative(image_id, bits)
    payload = run._canonical_json({"image_id": image_id, "bit_tuple": list(bits)})
    _append_cell_receipt(
        outdir=outdir,
        journal=journal,
        stage="benchmark-dev",
        prior_stage_seal=prior_stage_seal,
        destination=destination,
        payload=payload,
        purpose="persist unified primary/secondary resource schedules",
    )
    destination.unlink()

    exact_formats = {
        "sspl1": {
            "complete_bytes": 1,
            "blob_sha256": "1" * 64,
            "symbol_sha256": "2" * 64,
            "boundary": {"state_sha256": "3" * 64},
        },
        "ssp2l": {
            "complete_bytes": 1,
            "blob_sha256": "4" * 64,
            "symbol_sha256": "5" * 64,
            "boundary_state_sha256": "6" * 64,
        },
    }
    candidate_cell = {
        "image_id": image_id,
        "bit_tuple": list(bits),
        "relative_path": run._candidate_cell_relative(image_id, bits).as_posix(),
        "execution": {"exact_formats": exact_formats},
    }
    quality_cell = {
        "image_id": image_id,
        "bit_tuple": list(bits),
        "quality_cell_sha256": "7" * 64,
        "selection": {"state": "method_failure", "selected_label": None},
        "q_name": "sspl1",
        "formats": {},
        "primary_metrics": [],
        "variants": [],
    }
    preflight = {"native": {"arithmetic": {}}}
    monkeypatch.setattr(run, "_recover_interrupted_stage_completion", lambda *_args: None)
    monkeypatch.setattr(
        run,
        "_stage_guard",
        lambda _root, _stage: (journal, _NoOpGuard(), {"preflight": preflight}),
    )
    monkeypatch.setattr(
        run,
        "verify_candidate_stage",
        lambda _root: {"state": "complete_candidate_set", "cells": [candidate_cell]},
    )
    monkeypatch.setattr(
        run,
        "verify_quality_stage",
        lambda _root, **_kwargs: {
            "state": "complete_quality_grid",
            "quality_stage_sha256": prior_stage_seal,
            "cells": [quality_cell],
        },
    )
    monkeypatch.setattr(
        run,
        "_native_arithmetic_record",
        lambda _root, _record: (
            outdir / "runtime/native/arithmetic.so",
            {"path": "arithmetic.so", "bytes": 1, "sha256": "8" * 64},
        ),
    )
    decode_calls: list[str] = []

    def unexpected_decode(*_args: Any, **_kwargs: Any) -> Any:
        decode_calls.append("decode")
        raise UnexpectedWork("committed benchmark cell was measured again")

    monkeypatch.setattr(run, "_run_decode_schedule", unexpected_decode)

    with pytest.raises(run.LifecycleError, match="receipt|committed|missing|prefix|deleted"):
        run.benchmark_development(outdir)

    assert decode_calls == []
