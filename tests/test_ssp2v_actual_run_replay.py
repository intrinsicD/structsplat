"""Hostile, payload-free tests for the COMP-011 captured replay boundary.

These tests intentionally treat every persisted original candidate/quality record as adversarial.
Replay phase A must materialize and cold-decode regenerated scratch bytes; phase B may compare with
the originals, but must not inherit their paths or scalar decision inputs.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks import ssp2v_actual_coder as actual
from benchmarks import ssp2v_actual_run as run


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _repeat_metrics(psnr: float, ms_ssim: float, lpips: float) -> list[dict[str, float]]:
    return [
        {"psnr": psnr, "ms_ssim": ms_ssim, "lpips": lpips}
        for _ in range(3)
    ]


def test_replay_rejects_metric_exclusion_band_hidden_by_aggregate_failure() -> None:
    primary = _repeat_metrics(30.0, 0.9, 0.1)
    # PSNR definitely fails, so the aggregate class is `definitely_failed`,
    # while the 0.0005 MS-SSIM loss lies inside its frozen exclusion band.
    hidden_ambiguity = _repeat_metrics(29.94, 0.8995, 0.1)
    assert actual.evaluate_quality(primary, hidden_ambiguity)["state"] == "definitely_failed"
    with pytest.raises(run.LifecycleError, match="entered an exclusion band"):
        run._verify_replay_quality_classification(  # noqa: SLF001
            primary, hidden_ambiguity, primary, hidden_ambiguity
        )


def test_replay_accepts_failed_variant_when_every_metric_is_outside_its_band() -> None:
    primary = _repeat_metrics(30.0, 0.9, 0.1)
    outside = _repeat_metrics(29.94, 0.8996, 0.1)
    run._verify_replay_quality_classification(  # noqa: SLF001
        primary, outside, primary, outside
    )


def _materialize_captured_source_root(source_root: Path) -> None:
    """Create the smallest valid sealed-replica root needed by policy tests."""

    if source_root.exists():
        return
    source_root.mkdir(parents=True)
    manifest = actual.seal_record(
        {
            "schema": run.SOURCE_SCHEMA,
            "task_sha256": run.TASK_SHA256,
            "files": [],
        },
        "source_manifest_sha256",
    )
    manifest_path = source_root / "SOURCE_MANIFEST.json"
    manifest_path.write_bytes(run._canonical_json(manifest))
    manifest_path.chmod(0o444)
    source_root.chmod(0o555)
    assert run._worker_source_read_only(source_root) == [source_root]


def _captured_environment_authority(
    artifact_root: Path, source_root: Path | None = None
) -> dict[str, Any]:
    environment = run._worker_environment(artifact_root)
    if source_root is not None:
        _materialize_captured_source_root(source_root)
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                os.fspath(artifact_root / "runtime/metrics/site"),
                os.fspath(source_root),
                os.fspath(source_root / "src"),
            )
        )
        environment["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = str(
            run._verify_runtime_source_tree(source_root)["source_manifest_sha256"]
        )
    return {
        "sandboxed_workers": {
            "renderer_proof": {
                "launch": {
                    "worker_policy": {
                        "environment": environment
                    }
                }
            }
        }
    }


def _synthetic_replay_source_authority(
    artifact_root: Path,
    source_root: Path,
    *,
    launcher_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a sealed, payload-free replay authority for policy-only tests."""

    _materialize_captured_source_root(source_root)
    manifest = run._verify_runtime_source_tree(source_root)
    if launcher_sha256 is None:
        launcher_sha256 = run._sha256_file(
            Path(run.sandbox.__file__).resolve(strict=True)
        )
    launcher = {
        "path": "benchmarks/ssp2v_landlock.py",
        "bytes": 0,
        "sha256": launcher_sha256,
    }
    runtime_source = actual.seal_record(
        {"schema": "test-runtime-source-authority", "launcher": launcher},
        "runtime_source_sha256",
    )
    archive_sha256 = "a" * 64
    preflight = {
        **_captured_environment_authority(artifact_root, source_root),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "source_archive_sha256": archive_sha256,
        "runtime_source": runtime_source,
    }
    authority = actual.seal_record(
        {
            "schema": "structsplat.comp011.replay-captured-source-authority.v1",
            "source_root": os.fspath(source_root),
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "source_archive_sha256": archive_sha256,
            "runtime_source_sha256": runtime_source["runtime_source_sha256"],
            "launcher": launcher,
            "sealed_root_read_only": True,
            "transient_tree_reverified": True,
        },
        "replay_source_authority_sha256",
    )
    return preflight, authority


def _durable_replay_source_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Materialize the real archive/runtime receipts used by replay authority."""

    outdir = tmp_path / "artifact"
    outdir.mkdir()
    manifest = run.source_manifest(
        paths=(
            Path(run.__file__).resolve(strict=True),
            Path(run.sandbox.__file__).resolve(strict=True),
        )
    )
    (outdir / "source_manifest.json").write_bytes(run._canonical_json(manifest))
    archive = outdir / "source_snapshot.tar"
    run.write_source_archive(archive, manifest)
    archive_sha256 = run._sha256_file(archive)
    runtime_root = outdir / run._RUNTIME_SOURCE_RELATIVE
    run.safe_extract_source_archive(
        archive, runtime_root, expected_manifest=manifest
    )
    runtime_source = run._runtime_source_record(
        runtime_root,
        outdir=outdir,
        manifest=manifest,
        archive_sha256=archive_sha256,
    )
    preflight = {
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "source_archive_sha256": archive_sha256,
        "runtime_source": runtime_source,
        "sandboxed_workers": {
            "renderer_proof": {
                "launch": {
                    "worker_policy": {
                        "environment": run._worker_environment(outdir)
                    }
                }
            }
        },
    }
    return outdir, manifest, preflight


def _captured_decode_policy(
    *,
    command: list[str],
    artifact_root: Path,
    blob_relative_path: str,
    source_root: Path,
) -> dict[str, Any]:
    _materialize_captured_source_root(source_root)
    environment = run._worker_environment(artifact_root)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(artifact_root / "runtime/metrics/site"),
            str(source_root),
            str(source_root / "src"),
        )
    )
    environment["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = str(
        run._verify_runtime_source_tree(source_root)["source_manifest_sha256"]
    )
    return actual.seal_record(
        {
            "schema": run.WORKER_POLICY_SCHEMA,
            "command": command,
            "cwd": str(source_root / "benchmarks"),
            "environment": environment,
            "read_only_request": sorted(
                {
                    *(str(path) for path in run._worker_source_read_only(source_root)),
                    str(artifact_root / "runtime"),
                    str(artifact_root / blob_relative_path),
                }
            ),
            "read_write_request": [],
            "denied_read_probe_paths": [],
            "launcher_path": str(source_root / "benchmarks/ssp2v_landlock.py"),
            "launcher_sha256": run._sha256_file(
                Path(run.sandbox.__file__).resolve(strict=True)
            ),
            "timeout_seconds": 600.0,
        },
        "worker_policy_sha256",
    )


def _early_row(image_id: str, bits: tuple[int, int, int, int]) -> dict[str, Any]:
    return actual.seal_record(
        {
            "schema": actual.ROW_SCHEMA,
            "image_id": image_id,
            "bit_tuple": list(bits),
            "validity": "valid",
            "invalid_reason": None,
            "method_state": "lloyd_failure",
            "lloyd_all_fixed": False,
            "formats": None,
            "primary_metrics": None,
            "variants": None,
            "primary_resource": None,
            "secondary_resource": None,
        }
    )


def test_phase_a_cold_decodes_regenerated_scratch_bytes_not_persisted_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An equality check against the original is not a regenerated cold decode."""

    outdir = tmp_path / "artifact"
    scratch = tmp_path / "phase-a-scratch"
    source_path = outdir / "streams/source.sspl1"
    original_cell_root = outdir / "candidates/image/tuple"
    original_blob_path = original_cell_root / "exact/sspl1.bin"
    native_path = outdir / "runtime/native/arithmetic.so"
    lloyd_path = outdir / "runtime/native/lloyd.so"
    archive_path = outdir / "upstream/comp009.tar"
    for path, payload in (
        (source_path, b"sealed-source"),
        (original_blob_path, b"regenerated-complete-stream"),
        (native_path, b"native"),
        (lloyd_path, b"lloyd"),
        (archive_path, b"archive"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    scratch.mkdir()
    monkeypatch.setenv("COMP011_WORKER_SCRATCH", str(scratch))

    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    blob = original_blob_path.read_bytes()
    artifact = {
        "path": "exact/sspl1.bin",
        "bytes": len(blob),
        "sha256": _sha256(blob),
    }
    execution = {"status": "terminal_lloyd_failure"}
    cold_state = "c" * 64
    original_cell = {
        "image_id": image_id,
        "bit_tuple": list(bits),
        "state": "terminal_lloyd_failure",
        "relative_path": "candidates/image/tuple",
        "execution": execution,
        "artifacts": [artifact],
        "cold_decodes": {
            "exact/sspl1.bin": {
                "cold_seal": {"cold_decode_state_sha256": cold_state}
            }
        },
    }
    candidate = {
        "state": "valid_terminal_lloyd_failure",
        "cells": [original_cell],
        "candidate_set_sha256": None,
    }
    imported = {
        "streams": [
            {
                "image_id": image_id,
                "bit_tuple": list(bits),
                "relative_path": "streams/source.sspl1",
            }
        ]
    }
    preflight = {
        "native": {
            "lloyd": {"path": "lloyd.so", "bytes": 5, "sha256": _sha256(b"lloyd")}
        },
        "operational_upstream_roots": {"comp008": str(tmp_path / "comp008")},
    }
    authority = {
        "comp008": {"targets": [{"relative_path": "development/target.png"}]},
        "comp009": {
            "cells": [{"image_id": image_id, "bit_tuple": list(bits)}],
            "captured_source_archive": {"relative_path": "upstream/comp009.tar"},
            "captured_source_manifest": {},
            "source_manifest_sha256": "d" * 64,
        }
    }
    result = SimpleNamespace(
        record=execution,
        baseline_blobs={"sspl1": blob},
        variant_blobs={},
    )

    monkeypatch.setattr(run, "verify_preflight", lambda *_args, **_kwargs: preflight)
    monkeypatch.setattr(run, "verify_import_streams", lambda *_args, **_kwargs: imported)
    monkeypatch.setattr(run, "verify_candidate_stage", lambda *_args, **_kwargs: candidate)
    monkeypatch.setattr(run, "_authority_copy", lambda *_args, **_kwargs: authority)
    monkeypatch.setattr(
        run,
        "_native_arithmetic_record",
        lambda *_args, **_kwargs: (
            native_path,
            {"path": "arithmetic.so", "bytes": 6, "sha256": _sha256(b"native")},
        ),
    )
    monkeypatch.setattr(run, "_verify_regular_record", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run, "safe_extract_source_archive", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        run,
        "_run_producing_cell_worker",
        lambda **_kwargs: (result, {"producing_worker": {}}),
    )
    monkeypatch.setattr(run.execute, "verify_record", lambda _record: None)
    monkeypatch.setattr(
        run,
        "_cold_decode_expectations",
        lambda *_args, **_kwargs: ("1" * 64, "2" * 64),
    )
    monkeypatch.setattr(
        run.decode_worker,
        "stream_component_records",
        lambda value: {"total": {"bytes": len(value), "sha256": _sha256(value)}},
    )
    monkeypatch.setattr(
        run.decode_worker, "component_records_sha256", lambda _records: "3" * 64
    )

    decoded_paths: list[Path] = []

    def fake_cold_decode(**kwargs: Any) -> dict[str, Any]:
        path = Path(kwargs["blob_path"])
        decoded_paths.append(path)
        assert path.read_bytes() == blob
        return {
            "cold_seal": {"cold_decode_state_sha256": cold_state},
            "sandbox_attestation": {},
            "launch": {},
        }

    monkeypatch.setattr(run, "_run_cold_decode_sample", fake_cold_decode)

    replay = run._replay_candidate_set(outdir)

    assert replay["state"] == "valid_terminal_lloyd_failure"
    assert Path(replay["regenerated_artifact_root"]) == scratch / "regenerated-artifacts"
    assert decoded_paths
    assert all(path.is_relative_to(scratch) for path in decoded_paths)
    assert all(not path.is_relative_to(outdir / "candidates") for path in decoded_paths)


def test_phase_a_verifier_rejects_malformed_cold_seal_even_when_state_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The coarse state digest must not substitute for the full cold-seal schema verifier."""

    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    state = "4" * 64
    artifact = {"path": "exact/sspl1.bin", "bytes": 1, "sha256": _sha256(b"x")}
    execution = {
        "status": "terminal_lloyd_failure",
        "exact_formats": {
            "sspl1": {
                "complete_bytes": 1,
                "blob_sha256": _sha256(b"x"),
                "symbol_sha256": "5" * 64,
                "boundary": {"state_sha256": "6" * 64},
            }
        },
    }
    malformed = {"cold_decode_state_sha256": state}
    bundle = {
        "cold_seal": malformed,
        "sandbox_attestation": {},
        "launch": {"nonce": "7" * 64},
    }
    relative_path = run._candidate_cell_relative(image_id, bits).as_posix()
    outdir = tmp_path / "artifact"
    original_blob_path = outdir / relative_path / "exact/sspl1.bin"
    original_blob_path.parent.mkdir(parents=True)
    original_blob_path.write_bytes(b"x")
    original = {
        "image_id": image_id,
        "bit_tuple": list(bits),
        "state": "terminal_lloyd_failure",
        "relative_path": relative_path,
        "execution": execution,
        "artifacts": [artifact],
        "cold_decodes": {
            "exact/sspl1.bin": {"cold_seal": {"cold_decode_state_sha256": state}}
        },
    }
    producing_worker = actual.seal_record(
        {
            "execution": execution,
            "artifacts": [artifact],
            "diagnostics": {},
            "target_or_pixel_payloads_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "producing_worker_sha256",
    )
    regenerated = {
        **original,
        "cold_decodes": {"exact/sspl1.bin": bundle},
        "producing_diagnostics": {
            "producing_worker": {
                "worker": producing_worker,
                "sandbox_attestation": {},
                "launch": {"nonce": "3" * 64},
            }
        },
    }
    candidate = {
        "state": "valid_terminal_lloyd_failure",
        "cells": [original],
        "candidate_set_sha256": None,
    }
    phase_a = actual.seal_record(
        {
            "schema": run.REPLAY_PHASE_A_SCHEMA,
            "protocol": run.PROTOCOL,
            "task_sha256": run.TASK_SHA256,
            "source_manifest_sha256": "8" * 64,
            "candidate_replay": {
                "state": "valid_terminal_lloyd_failure",
                "candidate_set_sha256": None,
                "regenerated_artifact_root": str(
                    tmp_path / "worker-scratch/regenerated-artifacts"
                ),
                "cells": [regenerated],
                "cell_count": 1,
                "all_complete_streams_byte_identical": True,
                "all_cold_decode_states_identical": True,
                "targets_opened_before_candidate_match": False,
            },
            "targets_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "replay_phase_a_sha256",
    )
    monkeypatch.setattr(run, "_verify_launch_bundle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run.decode_worker,
        "stream_component_records",
        lambda value: {"total": {"bytes": len(value), "sha256": _sha256(value)}},
    )
    monkeypatch.setattr(
        run.decode_worker, "component_records_sha256", lambda _records: "9" * 64
    )

    calls: list[dict[str, Any]] = []

    def reject_malformed(record: Any, **kwargs: Any) -> None:
        calls.append({"record": record, **kwargs})
        raise run.decode_worker.DecodeWorkerError("malformed replay cold seal")

    monkeypatch.setattr(
        run.decode_worker, "verify_candidate_cold_seal", reject_malformed
    )

    with pytest.raises(run.decode_worker.DecodeWorkerError, match="malformed replay cold seal"):
        run._verify_replay_phase_a(
            outdir,
            phase_a,
            candidate,
            {
                "source_manifest_sha256": "8" * 64,
                "native": {
                    "arithmetic": {
                        "path": "arithmetic.so",
                        "bytes": 6,
                        "sha256": _sha256(b"native"),
                    }
                },
            },
            phase_a_sandbox_attestation_sha256="a" * 64,
        )
    assert len(calls) == 1


def test_cold_decode_policy_binds_regenerated_root_and_exact_command(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "captured-source"
    regenerated_root = tmp_path / "phase-a-scratch/regenerated-artifacts"
    original_root = tmp_path / "persisted-original"
    blob_relative = "candidates/image/tuple/exact/sspl1.bin"
    native_relative = "runtime/native/arithmetic.so"
    native = {"bytes": 6, "sha256": _sha256(b"native")}
    worker = run._decode_worker_source_record()
    command = run._candidate_cold_worker_command(
        artifact_root=regenerated_root,
        blob_relative_path=blob_relative,
        native_relative_path=native_relative,
        blob_sha256=_sha256(b"stream"),
        blob_bytes=6,
        native_record=native,
        worker_source=worker,
        symbol_sha256="1" * 64,
        boundary_state_sha256="2" * 64,
        stream_components_sha256="3" * 64,
    )
    valid_policy = _captured_decode_policy(
        command=command,
        artifact_root=regenerated_root,
        blob_relative_path=blob_relative,
        source_root=source_root,
    )
    preflight, replay_source_authority = _synthetic_replay_source_authority(
        regenerated_root, source_root
    )
    bundle = {"launch": {"nonce": "4" * 64, "worker_policy": valid_policy}}

    assert run._verify_captured_decode_worker_policy(
        bundle,
        expected_command=command,
        artifact_root=regenerated_root,
        blob_relative_path=blob_relative,
        native_relative_path=native_relative,
        expected_source_root=source_root,
        preflight_record=preflight,
        replay_source_authority=replay_source_authority,
    ) == source_root

    foreign_command = run._candidate_cold_worker_command(
        artifact_root=original_root,
        blob_relative_path=blob_relative,
        native_relative_path=native_relative,
        blob_sha256=_sha256(b"stream"),
        blob_bytes=6,
        native_record=native,
        worker_source=worker,
        symbol_sha256="1" * 64,
        boundary_state_sha256="2" * 64,
        stream_components_sha256="3" * 64,
    )
    foreign_policy = _captured_decode_policy(
        command=foreign_command,
        artifact_root=original_root,
        blob_relative_path=blob_relative,
        source_root=source_root,
    )
    foreign_bundle = {
        "launch": {"nonce": "5" * 64, "worker_policy": foreign_policy}
    }
    with pytest.raises(run.LifecycleError, match="persisted request drift"):
        run._verify_captured_decode_worker_policy(
            foreign_bundle,
            expected_command=command,
            artifact_root=regenerated_root,
            blob_relative_path=blob_relative,
            native_relative_path=native_relative,
            expected_source_root=source_root,
            preflight_record=preflight,
            replay_source_authority=replay_source_authority,
        )


def test_captured_decode_policy_accepts_every_frozen_system_read_only_path() -> None:
    rejected = [
        run._worker_policy_path(path)
        for path in run._sandbox_system_read_only()
        if not run._is_worker_system_readonly_path(run._worker_policy_path(path))
    ]
    assert rejected == []
    assert not run._is_worker_system_readonly_path("/tmp")


def test_replay_source_authority_rejects_mutation_and_survives_root_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use real archive receipts, then verify cold/producer policy after deletion."""

    outdir, manifest, preflight = _durable_replay_source_fixture(tmp_path)
    archive = outdir / "source_snapshot.tar"

    mutated_root = tmp_path / "mutated-workspace/randomized-source"
    run.safe_extract_source_archive(
        archive, mutated_root, expected_manifest=manifest
    )
    mutated_launcher = mutated_root / "benchmarks/ssp2v_landlock.py"
    payload = mutated_launcher.read_bytes()
    mutated_launcher.chmod(0o644)
    mutated_launcher.write_bytes(payload + b"\n")
    mutated_launcher.chmod(0o444)
    with pytest.raises(run.LifecycleError, match="differs from its manifest"):
        run._derive_replay_captured_source_authority(
            outdir, preflight, mutated_root
        )

    workspace = tmp_path / "verified-workspace"
    source_root = workspace / "randomized-source"
    run.safe_extract_source_archive(
        archive, source_root, expected_manifest=manifest
    )
    replay_source_authority = run._derive_replay_captured_source_authority(
        outdir, preflight, source_root
    )
    artifact_root = workspace / "phase-a-scratch/regenerated-artifacts"
    blob_relative = "candidates/image/tuple/exact/sspl1.bin"
    command = ["synthetic-post-deletion-cold-worker"]
    policy = _captured_decode_policy(
        command=command,
        artifact_root=artifact_root,
        blob_relative_path=blob_relative,
        source_root=source_root,
    )

    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    source = outdir / "streams" / image_id / run._tuple_label(bits) / "sspl1.bin"
    native = outdir / "runtime/native/arithmetic.so"
    lloyd = outdir / "runtime/native/lloyd.so"
    for path, payload in (
        (source, b"source-stream"),
        (native, b"native"),
        (lloyd, b"lloyd"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o444)
    native_record = {
        "path": native.name,
        "bytes": native.stat().st_size,
        "sha256": run._sha256_file(native),
    }
    lloyd_record = {
        "path": lloyd.name,
        "bytes": lloyd.stat().st_size,
        "sha256": run._sha256_file(lloyd),
    }
    producer_workspace = workspace / "phase-a-scratch/private-producer"
    captured_comp009 = workspace / "captured-comp009"
    producer_workspace.mkdir(parents=True)
    captured_comp009.mkdir()
    denied_probe = tmp_path / "protected-target.png"
    denied_probe.write_bytes(b"must remain unread")
    with monkeypatch.context() as source_context:
        source_context.setattr(
            run, "_worker_source_root", lambda _outdir: source_root
        )
        producer_policy = run._producer_launch_policy(
            outdir=outdir,
            source_path=source,
            captured_comp009_root=captured_comp009,
            native_path=native,
            lloyd_path=lloyd,
            config_path=producer_workspace / "config.json",
            output_root=producer_workspace / "output",
            denied_read_probes=[denied_probe],
        )
    comp009_authority = {
        "image_id": image_id,
        "bit_tuple": list(bits),
        "test_authority": True,
    }
    producer_config = actual.seal_record(
        {
            "schema": "structsplat.comp011.producing-cell-config.v1",
            "protocol": run.PROTOCOL,
            "task_sha256": run.TASK_SHA256,
            "image_id": image_id,
            "bit_tuple": list(bits),
            "source_bytes": source.stat().st_size,
            "source_sha256": run._sha256_file(source),
            "native_record": native_record,
            "lloyd_record": lloyd_record,
            "comp009_authority": comp009_authority,
            "captured_source_manifest_sha256": "c" * 64,
            "producer_policy": producer_policy,
        },
        "producing_config_sha256",
    )
    producer_worker = {
        "producing_config_sha256": producer_config["producing_config_sha256"],
        "producer_policy_sha256": producer_policy["producer_policy_sha256"],
        "artifact_root_identity": {
            "native": "runtime/native/arithmetic.so",
            "lloyd": "runtime/native/lloyd.so",
        },
    }
    nonce = "d" * 64
    attested_environment = dict(producer_policy["environment"])
    attested_environment["STRUCTSPLAT_PARENT_LAUNCH_NONCE"] = nonce
    child_pid = 424_242
    producer_attestation = {
        "pid": child_pid,
        "command": producer_policy["command"],
        "cwd": producer_policy["cwd"],
        "environment_keys": sorted(attested_environment),
        "environment_sha256_before_attestation_marker": run._sha256_bytes(
            run._canonical_json(attested_environment)
        ),
        "read_only": run._materialize_producer_request_paths(
            producer_policy["read_only_request"], child_pid=child_pid
        ),
        "read_write": run._materialize_producer_request_paths(
            producer_policy["read_write_request"], child_pid=child_pid
        ),
        "launcher_sha256": producer_policy["launcher_sha256"],
        "denied_read_probes": [
            {"path": os.fspath(denied_probe), "errno": errno.EACCES}
        ],
    }

    for directory in (
        source_root,
        *(path for path in source_root.rglob("*") if path.is_dir()),
    ):
        directory.chmod(0o755)
    shutil.rmtree(workspace)
    assert not source_root.exists()
    assert run._verify_captured_decode_worker_policy(
        {"launch": {"worker_policy": policy}},
        expected_command=command,
        artifact_root=artifact_root,
        blob_relative_path=blob_relative,
        native_relative_path="runtime/native/arithmetic.so",
        expected_source_root=source_root,
        preflight_record=preflight,
        replay_source_authority=replay_source_authority,
    ) == source_root
    monkeypatch.setattr(run, "_verify_launch_bundle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run, "_verify_sandbox_ancestry", lambda *_args, **_kwargs: None)
    run._verify_producer_launch_policy(
        outdir=outdir,
        image_id=image_id,
        bits=bits,
        config=producer_config,
        worker=producer_worker,
        attestation=producer_attestation,
        launch={"nonce": nonce},
        denied_read_probe_paths=[os.fspath(denied_probe)],
        expected_source_root=source_root,
        expected_native_record=native_record,
        expected_lloyd_record=lloyd_record,
        expected_comp009_authority=comp009_authority,
        expected_captured_source_manifest_sha256="c" * 64,
        expected_denied_read_probe_paths=[denied_probe],
        preflight_record=preflight,
        replay_source_authority=replay_source_authority,
        expected_parent_sandbox_attestation_sha256="f" * 64,
    )


@pytest.mark.parametrize(
    ("name", "hostile_value"),
    [
        ("CUDA_VISIBLE_DEVICES", "hostile-device-substitution"),
        ("LD_LIBRARY_PATH", "/tmp/hostile-loader-substitution"),
    ],
)
def test_cold_decode_policy_rejects_preflight_environment_substitution(
    tmp_path: Path,
    name: str,
    hostile_value: str,
) -> None:
    source_root = tmp_path / "captured-source"
    artifact_root = tmp_path / "phase-a-scratch/regenerated-artifacts"
    blob_relative = "candidates/image/tuple/exact/sspl1.bin"
    policy = _captured_decode_policy(
        command=["synthetic-cold-worker"],
        artifact_root=artifact_root,
        blob_relative_path=blob_relative,
        source_root=source_root,
    )
    preflight, replay_source_authority = _synthetic_replay_source_authority(
        artifact_root, source_root
    )
    hostile = copy.deepcopy(policy)
    hostile["environment"][name] = hostile_value
    hostile = _reseal_replay_worker_policy(hostile)

    with pytest.raises(run.LifecycleError, match="environment semantics drift"):
        run._verify_captured_decode_worker_policy(
            {"launch": {"worker_policy": hostile}},
            expected_command=["synthetic-cold-worker"],
            artifact_root=artifact_root,
            blob_relative_path=blob_relative,
            native_relative_path="runtime/native/arithmetic.so",
            expected_source_root=source_root,
            preflight_record=preflight,
            replay_source_authority=replay_source_authority,
        )


def test_resource_replay_policy_binds_blob_acquisition_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "captured-source"
    artifact_root = tmp_path / "phase-a-scratch/regenerated-artifacts"
    blob_relative = "candidates/image/tuple/exact/sspl1.bin"
    native_relative = "runtime/native/arithmetic.so"
    native = {
        "path": "arithmetic.so",
        "bytes": 6,
        "sha256": _sha256(b"native"),
    }
    worker = run._decode_worker_source_record()
    stream = {
        "blob_bytes": 6,
        "blob_sha256": _sha256(b"stream"),
        "symbol_sha256": "6" * 64,
        "boundary_state_sha256": "7" * 64,
    }
    launch = run.decode_worker.LaunchSpec("secondary", 0, "warmup", None, "sspl1")
    session = "8" * 64
    command = run._resource_worker_command(
        artifact_root=artifact_root,
        blob_relative_path=blob_relative,
        native_relative_path=native_relative,
        stream_record=stream,
        native_record=native,
        worker_source=worker,
        measurement_session_sha256=session,
        launch=launch,
    )
    policy = _captured_decode_policy(
        command=command,
        artifact_root=artifact_root,
        blob_relative_path=blob_relative,
        source_root=source_root,
    )
    preflight, replay_source_authority = _synthetic_replay_source_authority(
        artifact_root, source_root
    )
    preflight["native"] = {"arithmetic": native}
    expected_bindings = {
        "worker_module": {
            "source_label": run.decode_worker.WORKER_MODULE_LABEL,
            "bytes": worker["bytes"],
            "sha256": worker["sha256"],
            "identity_verified_before_operation": True,
        },
        "native_library": {
            "artifact_relative_path": native_relative,
            "bytes": native["bytes"],
            "sha256": native["sha256"],
            "identity_verified_from_same_opened_bytes_before_load": True,
            "loaded_from_sealed_verified_bytes": True,
        },
        "cdf_asset": {
            "source_label": "benchmarks/assets/ssp2e_cdf_v1.bin",
            "bytes": run.comp009_assets.CDF_ASSET_SIZE,
            "sha256": run.comp009_assets.CDF_ASSET_SHA256,
            "identity_verified_from_same_opened_bytes_before_load": True,
            "loaded_from_sealed_verified_bytes": True,
        },
    }
    sample = {
        **stream,
        "exactness": {
            "expected_blob_bytes": stream["blob_bytes"],
            "expected_symbol_sha256": stream["symbol_sha256"],
            "expected_boundary_state_sha256": stream["boundary_state_sha256"],
        },
        "blob_acquisition": {"artifact_relative_path": blob_relative},
        "source_bindings": expected_bindings,
    }
    schedule = {
        "schedule": [launch.to_dict()],
        "samples": [sample],
        "launches": [
            {"launch": {"nonce": "9" * 64, "worker_policy": policy}}
        ],
        "summary": {},
        "fresh_processes": True,
    }
    monkeypatch.setattr(run, "_verify_launch_bundle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run, "_verify_sandbox_ancestry", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run.decode_worker, "summarize_schedule", lambda *_args, **_kwargs: {})

    assert run._verify_replay_resource_schedule(
        schedule,
        frozen_schedule=[launch],
        image_index=0,
        tuple_index=0,
        measurement_session_sha256=session,
        expected_streams={"sspl1": stream},
        expected_blob_paths={"sspl1": blob_relative},
        artifact_root=artifact_root,
        expected_source_root=source_root,
        preflight_record=preflight,
        replay_source_authority=replay_source_authority,
        phase_b_sandbox_attestation_sha256="a" * 64,
    ) == ["9" * 64]

    sample["blob_acquisition"]["artifact_relative_path"] = "wrong/original.bin"
    with pytest.raises(run.LifecycleError, match="sample authority drift"):
        run._verify_replay_resource_schedule(
            schedule,
            frozen_schedule=[launch],
            image_index=0,
            tuple_index=0,
            measurement_session_sha256=session,
            expected_streams={"sspl1": stream},
            expected_blob_paths={"sspl1": blob_relative},
            artifact_root=artifact_root,
            expected_source_root=source_root,
            preflight_record=preflight,
            replay_source_authority=replay_source_authority,
            phase_b_sandbox_attestation_sha256="a" * 64,
        )


def test_phase_b_candidate_projection_does_not_inherit_original_only_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Original paths and arbitrary original fields must not flow into replay work."""

    outdir = tmp_path / "artifact"
    outdir.mkdir()
    phase_a_path = outdir / "phase_a.json"
    regenerated_root = tmp_path / "phase-a-scratch/regenerated-artifacts"
    regenerated_root.mkdir(parents=True)
    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    execution = {"status": "valid_complete_candidate_set"}
    regenerated_cell = {
        "image_id": image_id,
        "bit_tuple": list(bits),
        "state": "complete_candidate_cell",
        "relative_path": "phase-a-regenerated/cell",
        "execution": execution,
        "artifacts": [],
        "cold_decodes": {},
        "captured_comp009": {},
    }
    candidate_replay = {
        "state": "complete_candidate_set",
        "candidate_set_sha256": "9" * 64,
        "regenerated_artifact_root": str(regenerated_root),
        "cells": [regenerated_cell],
        "cell_count": 1,
    }
    phase_a = actual.seal_record(
        {
            "schema": run.REPLAY_PHASE_A_SCHEMA,
            "protocol": run.PROTOCOL,
            "task_sha256": run.TASK_SHA256,
            "source_manifest_sha256": "a" * 64,
            "candidate_replay": candidate_replay,
            "targets_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "replay_phase_a_sha256",
    )
    phase_a_path.write_bytes(run._canonical_json(phase_a))
    candidate = {
        "state": "complete_candidate_set",
        "candidate_set_sha256": "9" * 64,
        "cells": [
            {
                "image_id": image_id,
                "bit_tuple": list(bits),
                "state": "complete_candidate_cell",
                "relative_path": "candidates/persisted-original",
                "execution": {"status": "hostile-original"},
                "artifacts": [{"path": "hostile"}],
                "cold_decodes": {"hostile": True},
                "original_only_marker": "must-not-cross-replay-boundary",
            }
        ],
    }
    (outdir / "candidate_stage.json").write_bytes(run._canonical_json(candidate))
    (outdir / "analysis.json").write_bytes(run._canonical_json({}))

    monkeypatch.setattr(run, "source_manifest", lambda: {"source_manifest_sha256": "a" * 64})
    monkeypatch.setattr(run, "verify_target_copy", lambda *_args, **_kwargs: {"targets": []})
    monkeypatch.setattr(
        run,
        "verify_quality_stage",
        lambda *_args, **_kwargs: {"cells": [], "quality_stage_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        run,
        "verify_benchmark_stage",
        lambda *_args, **_kwargs: {"benchmark_stage_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        run,
        "verify_analysis_stage",
        lambda *_args, **_kwargs: {"analysis_record_sha256": "d" * 64},
    )
    monkeypatch.setattr(run, "verify_preflight", lambda *_args, **_kwargs: {})

    observed: dict[str, Any] = {}

    class CapturedProjection(RuntimeError):
        pass

    def capture_projection(_outdir: Path, *args: Any) -> Any:
        # The candidate is the third argument from the end whether the regenerated artifact root
        # has already been threaded into this helper or the older call shape is under test.
        replayed_candidate = args[-3]
        if len(args) == 5:
            observed["__candidate_artifact_root__"] = args[0]
        observed.update(replayed_candidate["cells"][0])
        raise CapturedProjection

    monkeypatch.setattr(run, "_replay_quality_grid", capture_projection)

    with pytest.raises(CapturedProjection):
        run.replay_phase_b_worker(outdir, phase_a_path)

    assert "original_only_marker" not in observed
    assert observed["relative_path"] == regenerated_cell["relative_path"]
    assert observed["execution"] == execution
    assert observed["__candidate_artifact_root__"] == regenerated_root


def test_terminal_phase_b_policy_never_stats_target_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal Lloyd failure never earns even target-path metadata capability."""

    outdir = tmp_path / "artifact"
    extracted = tmp_path / "randomized-source"
    phase_a_path = outdir / "replay/phase_a.json"
    extracted.mkdir()
    phase_a_path.parent.mkdir(parents=True)
    phase_a_path.write_bytes(b"{}")
    command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        run._CAPTURED_REPLAY_SCRIPT,
        str(extracted),
        "phase-b",
        str(outdir),
        str(phase_a_path),
    ]
    bundle = {
        "sandbox_attestation": {
            "pid": 12345,
            "command": command,
            "denied_read_probes": [],
        },
        "launch": {"nonce": "e" * 64},
    }
    preflight = {
        "source_manifest": {
            "files": [
                {
                    "path": "benchmarks/ssp2v_landlock.py",
                    "sha256": "f" * 64,
                }
            ]
        },
        "operational_upstream_roots": {"comp008": str(tmp_path / "comp008")},
    }
    candidate = {"state": "valid_terminal_lloyd_failure"}
    monkeypatch.setattr(run, "_verify_launch_bundle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run, "_sandbox_system_read_only", lambda: [])
    monkeypatch.setattr(run, "_sandbox_gpu_read_write", lambda: [])
    monkeypatch.setattr(
        run,
        "_authority_copy",
        lambda *_args, **_kwargs: {
            "comp008": {"targets": [{"relative_path": "development/target.png"}]}
        },
    )
    monkeypatch.setattr(run, "_attested_allowlist_path", lambda path, _pid: str(path))

    real_exists = Path.exists
    target_stats: list[Path] = []

    def observed_exists(path: Path) -> bool:
        if path == outdir / "targets" or path.is_relative_to(outdir / "targets"):
            target_stats.append(path)
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", observed_exists)
    try:
        run._verify_replay_outer_launch_policy(
            bundle=bundle,
            worker={},
            worker_seal_field="captured_replay_worker_sha256",
            phase="phase-b",
            outdir=outdir,
            phase_a_path=phase_a_path,
            preflight_record=preflight,
            candidate=candidate,
        )
    except run.LifecycleError:
        # The deliberately abbreviated attestation is expected to fail later; target stat is the
        # property under test.
        pass

    assert target_stats == []


def test_terminal_replay_stage_never_stats_or_allowlists_target_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launch policy itself must be target-free, not merely its later verifier."""

    outdir = tmp_path / "artifact"
    (outdir / "targets").mkdir(parents=True)
    preflight = {
        "preflight_binding_sha256": "b" * 64,
        "operational_upstream_roots": {"comp008": str(tmp_path / "comp008")},
    }
    candidate = {"state": "valid_terminal_lloyd_failure"}
    authority = {
        "comp008": {"targets": [{"relative_path": "development/target.png"}]},
    }

    monkeypatch.setattr(run, "_recover_interrupted_stage_completion", lambda *_args: None)
    monkeypatch.setattr(run, "verify_preflight", lambda *_args, **_kwargs: preflight)
    monkeypatch.setattr(run, "verify_import_streams", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run, "verify_candidate_stage", lambda *_args, **_kwargs: candidate)
    monkeypatch.setattr(run, "_authority_copy", lambda *_args, **_kwargs: authority)
    monkeypatch.setattr(run, "_read_canonical_object", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        run,
        "safe_extract_source_archive",
        lambda _archive, extracted, **_kwargs: _materialize_captured_source_root(
            extracted
        ),
    )
    monkeypatch.setattr(run, "_worker_environment", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run, "_sandbox_gpu_read_write", lambda: [])
    monkeypatch.setattr(run, "_verify_replay_phase_a", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run, "_verify_sandbox_ancestry", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run,
        "_derive_replay_captured_source_authority",
        lambda *_args, **_kwargs: {"test": "source-authority"},
    )
    monkeypatch.setattr(
        run,
        "_stage_guard",
        lambda *_args, **_kwargs: (
            object(),
            object(),
            {"analyze": {"analysis_record_sha256": "0" * 64}},
        ),
    )
    monkeypatch.setattr(
        run,
        "_journaled_immutable",
        lambda *_args, **_kwargs: {"relative_path": "replay/phase_a.json"},
    )
    monkeypatch.setattr(
        run,
        "_artifact_path",
        lambda root, relative, *_args, **_kwargs: root / relative,
    )

    real_exists = Path.exists
    target_stats: list[Path] = []

    def observed_exists(path: Path) -> bool:
        if path == outdir / "targets" or path.is_relative_to(outdir / "targets"):
            target_stats.append(path)
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", observed_exists)
    phase_b_readonly: list[Path] = []

    class CapturedPhaseB(RuntimeError):
        pass

    def capture_launch(command: list[str], **kwargs: Any) -> Any:
        if "phase-a" in command:
            return {}, {"attestation_sha256": "a" * 64}, {}
        phase_b_readonly.extend(kwargs["read_only"])
        raise CapturedPhaseB

    monkeypatch.setattr(run, "_run_json_worker", capture_launch)

    with pytest.raises(CapturedPhaseB):
        run.replay_stage(outdir)

    assert target_stats == []
    assert all(
        path != outdir / "targets" and not path.is_relative_to(outdir / "targets")
        for path in phase_b_readonly
    )


def test_terminal_phase_b_persists_full_replay_rows_and_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate signature alone is not replay evidence, even on the early-failure branch."""

    outdir = tmp_path / "artifact"
    outdir.mkdir()
    phase_a_path = outdir / "phase_a.json"
    regenerated_root = tmp_path / "terminal-scratch/regenerated-artifacts"
    regenerated_root.mkdir(parents=True)
    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    candidate_cell = {
        "image_id": image_id,
        "bit_tuple": list(bits),
        "state": "terminal_lloyd_failure",
        "relative_path": run._candidate_cell_relative(image_id, bits).as_posix(),
        "execution": {"status": "terminal_lloyd_failure"},
        "artifacts": [],
        "cold_decodes": {},
    }
    candidate_replay = {
        "state": "valid_terminal_lloyd_failure",
        "candidate_set_sha256": None,
        "regenerated_artifact_root": str(regenerated_root),
        "cells": [candidate_cell],
        "cell_count": 1,
    }
    phase_a = actual.seal_record(
        {
            "schema": run.REPLAY_PHASE_A_SCHEMA,
            "protocol": run.PROTOCOL,
            "task_sha256": run.TASK_SHA256,
            "source_manifest_sha256": "1" * 64,
            "candidate_replay": candidate_replay,
            "targets_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "replay_phase_a_sha256",
    )
    phase_a_path.write_bytes(run._canonical_json(phase_a))
    (outdir / "candidate_stage.json").write_bytes(
        run._canonical_json(
            {
                "state": "valid_terminal_lloyd_failure",
                "candidate_set_sha256": None,
                "cells": [candidate_cell],
            }
        )
    )
    early_row = _early_row(image_id, bits)
    analysis = actual.analyze_rows([early_row])
    (outdir / "analysis.json").write_bytes(run._canonical_json(analysis))
    (outdir / "analysis_record.json").write_bytes(
        run._canonical_json({"analysis_record_sha256": "2" * 64})
    )
    benchmark = actual.seal_record(
        {"schema": "test-terminal-benchmark"}, "benchmark_stage_sha256"
    )
    (outdir / "benchmark_stage.json").write_bytes(run._canonical_json(benchmark))
    monkeypatch.setattr(run, "source_manifest", lambda: {"source_manifest_sha256": "1" * 64})

    replay = run.replay_phase_b_worker(outdir, phase_a_path)

    assert replay["replay_rows"] == [early_row]
    assert replay["replay_analysis"] == analysis
    assert replay["analysis_gate_signature"] == run._analysis_gate_signature(analysis)


def test_gate_signature_collision_demonstrates_need_for_full_replay_evidence() -> None:
    """Document that equal gate projections do not prove equal exact rates."""

    gates = {
        "geometric_mean_lte_0_95_exact": True,
        "strict_image_wins_gte_7": True,
    }
    first = {
        "status": "complete",
        "decision": "same",
        "ssp2l_decision": "same",
        "tuple_results": [
            {
                "state": "valid",
                "bit_tuple": list(actual.FROZEN_TUPLES[0]),
                "candidate_bytes": [90] * 8,
                "primary_bytes": [100] * 8,
                "actual_rate": {"gates": gates, "pass": True},
                "all_gates_pass": True,
            }
        ],
        "secondary_tuple_results": [],
    }
    second = {
        **first,
        "tuple_results": [
            {
                **first["tuple_results"][0],
                "candidate_bytes": [1] * 8,
                "primary_bytes": [10_000] * 8,
            }
        ],
    }

    assert first != second
    assert run._analysis_gate_signature(first) == run._analysis_gate_signature(second)


def _reseal_replay_worker_policy(policy: dict[str, Any]) -> dict[str, Any]:
    material = copy.deepcopy(policy)
    material.pop("worker_policy_sha256", None)
    return actual.seal_record(material, "worker_policy_sha256")


def _captured_replay_environment(
    *,
    outdir: Path,
    captured_source_root: Path,
    tmpdir: Path | None = None,
    worker_scratch: Path | None = None,
    preflight_binding_sha256: str | None = None,
) -> dict[str, str]:
    environment = run._worker_environment(outdir)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            os.fspath(outdir / "runtime/metrics/site"),
            os.fspath(captured_source_root),
            os.fspath(captured_source_root / "src"),
        )
    )
    if (captured_source_root / "SOURCE_MANIFEST.json").is_file():
        environment["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = str(
            run._verify_runtime_source_tree(captured_source_root)[
                "source_manifest_sha256"
            ]
        )
    if tmpdir is not None:
        environment["TMPDIR"] = os.fspath(tmpdir)
    if worker_scratch is not None:
        environment["COMP011_WORKER_SCRATCH"] = os.fspath(worker_scratch)
    if preflight_binding_sha256 is not None:
        environment[run._REPLAY_PREFLIGHT_BINDING_ENV] = preflight_binding_sha256
        environment.update(run._replay_path_inventory_environment())  # noqa: SLF001
    return environment


def _render_metric_policy_fixture(tmp_path: Path) -> dict[str, Any]:
    outdir = tmp_path / "artifact"
    workspace = tmp_path / "replay-workspace"
    captured_source_root = workspace / "captured-source"
    _materialize_captured_source_root(captured_source_root)
    regenerated_root = workspace / "phase-a-scratch/regenerated-artifacts"
    render_temp = workspace / "phase-b-scratch/comp011-render-metric-authority"
    render_path = render_temp / "render.npy"
    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    candidate_cell = {
        "image_id": image_id,
        "bit_tuple": list(bits),
        "relative_path": run._candidate_cell_relative(image_id, bits).as_posix(),
        "execution": {"selected_q": {"name": "ssp2v"}},
    }
    blob_path = (
        regenerated_root / candidate_cell["relative_path"] / "exact/ssp2v.bin"
    )
    native = {
        "path": "arithmetic.so",
        "bytes": 6,
        "sha256": _sha256(b"native"),
    }
    expected_stream = {
        "blob_sha256": _sha256(b"regenerated-stream"),
        "symbol_sha256": "1" * 64,
        "boundary_state_sha256": "2" * 64,
    }
    render = {
        "render_npy": {"sha256": _sha256(b"render-npy")},
        "render_tensor": {"sha256": _sha256(b"render-tensor")},
    }
    target_record = {
        "rgb_copy": {
            "relative_path": f"targets/{image_id}/target.rgb",
            "sha256": _sha256(b"target-rgb"),
        },
        "rgb_dimensions_hw": [8, 9],
    }
    native_path = outdir / "runtime/native/arithmetic.so"
    renderer_path = outdir / "runtime/renderer/record.json"
    target_path = outdir / target_record["rgb_copy"]["relative_path"]
    launcher_path = captured_source_root / "benchmarks/ssp2v_landlock.py"
    launcher_sha256 = "3" * 64
    render_command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_actual_run",
        "_render-arm",
        "--artifact-root",
        os.fspath(outdir),
        "--renderer-record",
        os.fspath(renderer_path),
        "--blob",
        os.fspath(blob_path),
        "--native",
        os.fspath(native_path),
        "--expected-blob-sha256",
        expected_stream["blob_sha256"],
        "--expected-native-sha256",
        native["sha256"],
        "--expected-native-bytes",
        str(native["bytes"]),
        "--expected-symbol-sha256",
        expected_stream["symbol_sha256"],
        "--expected-boundary-state-sha256",
        expected_stream["boundary_state_sha256"],
        "--output",
        os.fspath(render_path),
    ]
    metric_command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_actual_run",
        "_metric-worker",
        "--render",
        os.fspath(render_path),
        "--target-rgb",
        os.fspath(target_path),
        "--height",
        str(target_record["rgb_dimensions_hw"][0]),
        "--width",
        str(target_record["rgb_dimensions_hw"][1]),
        "--expected-render-npy-sha256",
        render["render_npy"]["sha256"],
        "--expected-render-tensor-sha256",
        render["render_tensor"]["sha256"],
        "--expected-target-rgb-sha256",
        target_record["rgb_copy"]["sha256"],
        "--expected-lpips-state-sha256",
        "4" * 64,
    ]
    render_policy = _reseal_replay_worker_policy(
        {
            "schema": run.WORKER_POLICY_SCHEMA,
            "command": render_command,
            "cwd": os.fspath(captured_source_root / "benchmarks"),
            "environment": _captured_replay_environment(
                outdir=outdir,
                captured_source_root=captured_source_root,
                tmpdir=render_temp,
            ),
            "read_only_request": sorted(
                os.fspath(path)
                for path in (
                    *run._worker_source_read_only(captured_source_root),
                    outdir / "runtime",
                    blob_path,
                )
            ),
            "read_write_request": [os.fspath(render_temp)],
            "denied_read_probe_paths": [],
            "launcher_path": os.fspath(launcher_path),
            "launcher_sha256": launcher_sha256,
            "timeout_seconds": 600.0,
        }
    )
    metric_policy = _reseal_replay_worker_policy(
        {
            "schema": run.WORKER_POLICY_SCHEMA,
            "command": metric_command,
            "cwd": os.fspath(captured_source_root / "benchmarks"),
            "environment": _captured_replay_environment(
                outdir=outdir,
                captured_source_root=captured_source_root,
                tmpdir=render_temp / "metric-scratch",
            ),
            "read_only_request": sorted(
                os.fspath(path)
                for path in (
                    *run._worker_source_read_only(captured_source_root),
                    outdir / "runtime",
                    render_path,
                    target_path,
                )
            ),
            "read_write_request": [
                os.fspath(render_temp / "metric-scratch")
            ],
            "denied_read_probe_paths": [],
            "launcher_path": os.fspath(launcher_path),
            "launcher_sha256": launcher_sha256,
            "timeout_seconds": 600.0,
        }
    )
    preflight, replay_source_authority = _synthetic_replay_source_authority(
        outdir,
        captured_source_root,
        launcher_sha256=launcher_sha256,
    )
    preflight.update(
        {
            "native": {"arithmetic": native},
            "environment": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
            },
            "renderer": {
                "abi": {
                    "loader_environment": {
                        "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
                        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
                    }
                }
            },
            "operational_upstream_roots": {
                "comp008": os.fspath(tmp_path / "protected-comp008")
            },
        }
    )
    return {
        "task": {
            "render_launch": {"worker_policy": render_policy},
            "metric_launch": {"worker_policy": metric_policy},
        },
        "outdir": outdir,
        "workspace": workspace,
        "captured_source_root": captured_source_root,
        "regenerated_root": regenerated_root,
        "candidate_cell": candidate_cell,
        "expected_stream": expected_stream,
        "render": render,
        "target_record": target_record,
        "preflight": preflight,
        "replay_source_authority": replay_source_authority,
        "lpips_state": "4" * 64,
        "blob_path": blob_path,
        "render_path": render_path,
        "target_path": target_path,
    }


def _verify_render_metric_fixture(fixture: dict[str, Any]) -> None:
    run._verify_replay_render_metric_policy(
        fixture["task"],
        outdir=fixture["outdir"],
        captured_source_root=fixture["captured_source_root"],
        regenerated_artifact_root=fixture["regenerated_root"],
        candidate_cell=fixture["candidate_cell"],
        stream_name="ssp2v",
        expected_stream=fixture["expected_stream"],
        render=fixture["render"],
        target_record=fixture["target_record"],
        preflight_record=fixture["preflight"],
        replay_source_authority=fixture["replay_source_authority"],
        expected_lpips_state_sha256=fixture["lpips_state"],
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "render_blob",
        "source_root",
        "render_environment",
        "render_cuda_device_environment",
        "render_loader_environment",
        "render_cwd",
        "render_read_only",
        "render_read_write",
        "render_launcher",
        "render_denied",
        "render_timeout",
        "metric_render_link",
        "metric_target_link",
        "metric_environment",
        "metric_cwd",
        "metric_read_only",
        "metric_read_write",
        "metric_launcher",
        "metric_denied",
        "metric_timeout",
    ],
)
def test_replay_render_metric_policy_rejects_coherently_resealed_substitution(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Every data/control capability remains bound after a valid policy is re-sealed."""

    fixture = _render_metric_policy_fixture(tmp_path)
    _verify_render_metric_fixture(fixture)
    task = copy.deepcopy(fixture["task"])
    render = task["render_launch"]["worker_policy"]
    metric = task["metric_launch"]["worker_policy"]
    foreign_source = fixture["workspace"] / "foreign-source"
    foreign_render = fixture["workspace"] / "phase-b-scratch/foreign/render.npy"
    foreign_target = fixture["outdir"] / "targets/foreign/target.rgb"

    if mutation == "render_blob":
        original = os.fspath(fixture["blob_path"])
        foreign = os.fspath(
            fixture["outdir"]
            / fixture["candidate_cell"]["relative_path"]
            / "exact/ssp2v.bin"
        )
        render["command"][render["command"].index("--blob") + 1] = foreign
        render["read_only_request"].remove(original)
        render["read_only_request"].append(foreign)
    elif mutation == "source_root":
        for policy in (render, metric):
            old = fixture["captured_source_root"]
            policy["cwd"] = os.fspath(foreign_source / "benchmarks")
            policy["environment"]["PYTHONPATH"] = os.pathsep.join(
                (
                    os.fspath(fixture["outdir"] / "runtime/metrics/site"),
                    os.fspath(foreign_source),
                    os.fspath(foreign_source / "src"),
                )
            )
            policy["read_only_request"] = [
                value.replace(os.fspath(old), os.fspath(foreign_source))
                for value in policy["read_only_request"]
            ]
            policy["launcher_path"] = os.fspath(
                foreign_source / "benchmarks/ssp2v_landlock.py"
            )
    elif mutation == "render_environment":
        render["environment"]["PYTHONPATH"] = os.fspath(foreign_source)
    elif mutation == "render_cuda_device_environment":
        render["environment"]["CUDA_VISIBLE_DEVICES"] = "hostile-device-substitution"
    elif mutation == "render_loader_environment":
        render["environment"]["LD_LIBRARY_PATH"] = os.fspath(
            fixture["captured_source_root"] / "hostile-loader-path"
        )
    elif mutation == "render_cwd":
        render["cwd"] = os.fspath(foreign_source / "benchmarks")
    elif mutation == "render_read_only":
        render["read_only_request"].append(os.fspath(fixture["outdir"] / "targets"))
    elif mutation == "render_read_write":
        render["read_write_request"].append(
            os.fspath(fixture["outdir"] / "candidates")
        )
    elif mutation == "render_launcher":
        render["launcher_path"] = os.fspath(
            foreign_source / "benchmarks/ssp2v_landlock.py"
        )
    elif mutation == "render_denied":
        render["denied_read_probe_paths"] = [
            os.fspath(fixture["outdir"] / "targets/secret.rgb")
        ]
    elif mutation == "render_timeout":
        render["timeout_seconds"] = 601.0
    elif mutation == "metric_render_link":
        original = os.fspath(fixture["render_path"])
        foreign = os.fspath(foreign_render)
        metric["command"][metric["command"].index("--render") + 1] = foreign
        metric["read_only_request"].remove(original)
        metric["read_only_request"].append(foreign)
    elif mutation == "metric_target_link":
        original = os.fspath(fixture["target_path"])
        foreign = os.fspath(foreign_target)
        metric["command"][metric["command"].index("--target-rgb") + 1] = foreign
        metric["read_only_request"].remove(original)
        metric["read_only_request"].append(foreign)
    elif mutation == "metric_environment":
        metric["environment"]["PYTHONPATH"] = os.fspath(foreign_source)
    elif mutation == "metric_cwd":
        metric["cwd"] = os.fspath(foreign_source / "benchmarks")
    elif mutation == "metric_read_only":
        metric["read_only_request"].append(
            os.fspath(fixture["outdir"] / "candidates")
        )
    elif mutation == "metric_read_write":
        metric["read_write_request"].append(
            os.fspath(fixture["workspace"] / "phase-b-scratch/foreign-write")
        )
    elif mutation == "metric_launcher":
        metric["launcher_path"] = os.fspath(
            foreign_source / "benchmarks/ssp2v_landlock.py"
        )
    elif mutation == "metric_denied":
        metric["denied_read_probe_paths"] = [
            os.fspath(fixture["outdir"] / "targets/secret.rgb")
        ]
    elif mutation == "metric_timeout":
        metric["timeout_seconds"] = 601.0
    else:  # pragma: no cover - the parameter inventory above is explicit
        raise AssertionError(f"unknown mutation {mutation}")

    task["render_launch"]["worker_policy"] = _reseal_replay_worker_policy(render)
    task["metric_launch"]["worker_policy"] = _reseal_replay_worker_policy(metric)
    fixture["task"] = task
    with pytest.raises(run.LifecycleError):
        _verify_render_metric_fixture(fixture)


def _outer_replay_policy_fixture(tmp_path: Path) -> dict[str, Any]:
    outdir = tmp_path / "artifact"
    workspace = tmp_path / "replay-workspace"
    captured_source_root = workspace / "captured-source"
    phase_a_path = outdir / "replay/phase_a.json"
    scratch = workspace / "phase-b-scratch"
    replay_scratch = workspace / "phase-a-scratch"
    regenerated_root = replay_scratch / "regenerated-artifacts"
    command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        run._CAPTURED_REPLAY_SCRIPT,
        os.fspath(captured_source_root),
        "phase-b",
        os.fspath(outdir),
        os.fspath(phase_a_path),
    ]
    explicit_read_only = [
        captured_source_root,
        outdir / "runtime",
        outdir / "upstream",
        outdir / "streams",
        outdir / "candidates",
        outdir / "targets",
        outdir / "quality",
        outdir / "benchmark",
        outdir / "journal/events",
        outdir / "preflight.json",
        outdir / "source_manifest.json",
        outdir / "source_snapshot.tar",
        outdir / "import_streams.json",
        outdir / "candidate_stage.json",
        outdir / "target_copy.json",
        outdir / "quality_stage.json",
        outdir / "benchmark_stage.json",
        outdir / "actual_rows.jsonl",
        outdir / "analysis.json",
        outdir / "analysis_record.json",
        phase_a_path,
        replay_scratch,
    ]
    upstream_root = tmp_path / "protected-comp008"
    upstream_target = upstream_root / "development/target.png"
    launcher_sha256 = "a" * 64
    preflight_binding_sha256 = "b" * 64
    replay_environment = _captured_replay_environment(
        outdir=outdir,
        captured_source_root=captured_source_root,
        tmpdir=scratch,
        worker_scratch=scratch,
        preflight_binding_sha256=preflight_binding_sha256,
    )
    system_read_only = json.loads(
        replay_environment[run._REPLAY_SYSTEM_READ_ONLY_ENV]  # noqa: SLF001
    )
    gpu_read_write = json.loads(
        replay_environment[run._REPLAY_GPU_READ_WRITE_ENV]  # noqa: SLF001
    )
    policy = _reseal_replay_worker_policy(
        {
            "schema": run.WORKER_POLICY_SCHEMA,
            "command": command,
            "cwd": os.fspath(captured_source_root),
            "environment": replay_environment,
            "read_only_request": sorted(
                {*system_read_only, *(os.fspath(path) for path in explicit_read_only)}
            ),
            "read_write_request": sorted(
                {*gpu_read_write, os.fspath(scratch)}
            ),
            "denied_read_probe_paths": [os.fspath(upstream_target)],
            "launcher_path": os.fspath(
                captured_source_root / "benchmarks/ssp2v_landlock.py"
            ),
            "launcher_sha256": launcher_sha256,
            "timeout_seconds": 24.0 * 3600.0,
        }
    )
    return {
        "bundle": {
            "sandbox_attestation": {"command": command},
            "launch": {"worker_policy": policy},
        },
        "worker": {
            "candidate_replay": {
                "regenerated_artifact_root": os.fspath(regenerated_root)
            }
        },
        "outdir": outdir,
        "workspace": workspace,
        "captured_source_root": captured_source_root,
        "phase_a_path": phase_a_path,
        "candidate": {"state": "complete_candidate_set"},
        "preflight": {
            "preflight_binding_sha256": preflight_binding_sha256,
            "environment": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
            },
            "renderer": {
                "abi": {
                    "loader_environment": {
                        "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
                        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
                    }
                }
            },
            "source_manifest": {
                "files": [
                    {
                        "path": "benchmarks/ssp2v_landlock.py",
                        "sha256": launcher_sha256,
                    }
                ]
            },
            "operational_upstream_roots": {"comp008": os.fspath(upstream_root)},
        },
        "authority": {
            "comp008": {
                "targets": [{"relative_path": "development/target.png"}]
            }
        },
    }


def _verify_outer_replay_fixture(
    fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "_verify_launch_bundle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run, "_verify_sandbox_ancestry", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run, "_authority_copy", lambda *_args, **_kwargs: fixture["authority"]
    )
    run._verify_replay_outer_launch_policy(
        bundle=fixture["bundle"],
        worker=fixture["worker"],
        worker_seal_field="captured_replay_worker_sha256",
        phase="phase-b",
        outdir=fixture["outdir"],
        phase_a_path=fixture["phase_a_path"],
        preflight_record=fixture["preflight"],
        candidate=fixture["candidate"],
        expected_captured_source_root=fixture["captured_source_root"],
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "outer_command",
        "outer_source_root",
        "outer_environment",
        "outer_inventory_missing",
        "outer_inventory_broad",
        "outer_inventory_policy_missing",
        "outer_preflight_binding_missing",
        "outer_preflight_binding_wrong",
        "outer_cwd",
        "outer_read_only",
        "outer_read_only_alias",
        "outer_read_only_broad_proc",
        "outer_read_only_broad_sys",
        "outer_read_only_cpu_tree",
        "outer_read_write",
        "outer_launcher",
        "outer_denied",
        "outer_timeout",
        "regenerated_root",
    ],
)
def test_replay_outer_policy_rejects_resealed_root_and_capability_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Phase B stays tied to phase A's root, scratch, policy, and regenerated bytes."""

    fixture = _outer_replay_policy_fixture(tmp_path)
    _verify_outer_replay_fixture(fixture, monkeypatch)
    bundle = copy.deepcopy(fixture["bundle"])
    policy = bundle["launch"]["worker_policy"]
    foreign_source = fixture["workspace"] / "foreign-source"
    foreign_scratch = fixture["workspace"] / "foreign-phase-b-scratch"

    if mutation == "outer_command":
        policy["command"][-1] = os.fspath(
            fixture["outdir"] / "replay/foreign-phase-a.json"
        )
    elif mutation == "outer_source_root":
        bundle["sandbox_attestation"]["command"][5] = os.fspath(foreign_source)
        policy["command"][5] = os.fspath(foreign_source)
        policy["cwd"] = os.fspath(foreign_source)
        policy["environment"] = _captured_replay_environment(
            outdir=fixture["outdir"],
            captured_source_root=foreign_source,
            tmpdir=foreign_scratch,
            worker_scratch=foreign_scratch,
            preflight_binding_sha256=fixture["preflight"][
                "preflight_binding_sha256"
            ],
        )
        policy["read_only_request"] = [
            value.replace(
                os.fspath(fixture["captured_source_root"]), os.fspath(foreign_source)
            )
            for value in policy["read_only_request"]
        ]
        policy["read_write_request"] = [os.fspath(foreign_scratch)]
        policy["launcher_path"] = os.fspath(
            foreign_source / "benchmarks/ssp2v_landlock.py"
        )
    elif mutation == "outer_environment":
        policy["environment"]["PYTHONPATH"] = os.fspath(foreign_source)
    elif mutation == "outer_inventory_missing":
        policy["environment"].pop(run._REPLAY_SYSTEM_READ_ONLY_ENV)  # noqa: SLF001
    elif mutation == "outer_inventory_broad":
        values = json.loads(
            policy["environment"][run._REPLAY_SYSTEM_READ_ONLY_ENV]  # noqa: SLF001
        )
        policy["environment"][run._REPLAY_SYSTEM_READ_ONLY_ENV] = (  # noqa: SLF001
            run._canonical_json(sorted({*values, "/sys"}))  # noqa: SLF001
            .decode()
            .removesuffix("\n")
        )
    elif mutation == "outer_inventory_policy_missing":
        policy["read_only_request"].remove("/usr")
    elif mutation == "outer_preflight_binding_missing":
        policy["environment"].pop(run._REPLAY_PREFLIGHT_BINDING_ENV)
    elif mutation == "outer_preflight_binding_wrong":
        policy["environment"][run._REPLAY_PREFLIGHT_BINDING_ENV] = "c" * 64
    elif mutation == "outer_cwd":
        policy["cwd"] = os.fspath(foreign_source)
    elif mutation == "outer_read_only":
        policy["read_only_request"].append(
            os.fspath(Path(fixture["preflight"]["operational_upstream_roots"]["comp008"]))
        )
    elif mutation == "outer_read_only_alias":
        policy["read_only_request"][0] = (
            policy["read_only_request"][0] + "/../" + Path(policy["read_only_request"][0]).name
        )
    elif mutation == "outer_read_only_broad_proc":
        policy["read_only_request"].append("/proc")
    elif mutation == "outer_read_only_broad_sys":
        policy["read_only_request"].append("/sys")
    elif mutation == "outer_read_only_cpu_tree":
        policy["read_only_request"].append("/sys/devices/system/cpu")
    elif mutation == "outer_read_write":
        policy["read_write_request"].append(
            os.fspath(fixture["outdir"] / "targets")
        )
    elif mutation == "outer_launcher":
        policy["launcher_path"] = os.fspath(
            foreign_source / "benchmarks/ssp2v_landlock.py"
        )
    elif mutation == "outer_denied":
        policy["denied_read_probe_paths"] = []
    elif mutation == "outer_timeout":
        policy["timeout_seconds"] = 60.0
    elif mutation == "regenerated_root":
        fixture["worker"] = {
            "candidate_replay": {
                "regenerated_artifact_root": os.fspath(
                    fixture["workspace"] / "foreign-regenerated-artifacts"
                )
            }
        }
    else:  # pragma: no cover - the parameter inventory above is explicit
        raise AssertionError(f"unknown mutation {mutation}")

    bundle["launch"]["worker_policy"] = _reseal_replay_worker_policy(policy)
    fixture["bundle"] = bundle
    with pytest.raises(run.LifecycleError):
        _verify_outer_replay_fixture(fixture, monkeypatch)
