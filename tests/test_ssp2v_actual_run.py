from __future__ import annotations

import hashlib
import io
import os
import stat
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks import ssp2v_actual_coder as actual
from benchmarks import ssp2v_actual_run as run


def test_strict_png_samples_use_comp008_shape_prefixed_pixel_authority() -> None:
    from PIL import Image

    from benchmarks import sgi_entropy_oracle_run as comp008_run

    pixels = np.asarray(
        [
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[255, 254, 253], [252, 251, 250], [17, 33, 65]],
        ],
        dtype=np.uint8,
    )
    with io.BytesIO() as handle:
        Image.fromarray(pixels, mode="RGB").save(handle, format="PNG")
        payload = handle.getvalue()
    authority = {
        "rgb_dimensions_hw": [2, 3],
        "rgb_pixel_sha256": run._comp008_rgb_pixel_sha256(pixels),  # noqa: SLF001
    }
    assert authority["rgb_pixel_sha256"] == comp008_run._pixel_sha256(pixels)  # noqa: SLF001

    decoded, record = run._strict_png_samples(payload, authority)  # noqa: SLF001

    assert np.array_equal(decoded, pixels)
    assert record["rgb_pixel_sha256"] == authority["rgb_pixel_sha256"]
    raw_only = hashlib.sha256(pixels.tobytes(order="C")).hexdigest()
    assert raw_only != authority["rgb_pixel_sha256"]
    with pytest.raises(run.LifecycleError, match="strict-RGB authority"):
        run._strict_png_samples(  # noqa: SLF001
            payload,
            {**authority, "rgb_pixel_sha256": raw_only},
        )


def test_metric_dependency_warning_audit_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warnings

    sentinel = object()

    def expected(_render: object, _target: object) -> tuple[object, str]:
        for category_name, message in run._EXPECTED_METRIC_DEPENDENCY_WARNINGS:  # noqa: SLF001
            assert category_name == "UserWarning"
            warnings.warn(message, UserWarning, stacklevel=1)
        return sentinel, "a" * 64

    monkeypatch.setattr(run.quality, "exact_metric_values", expected)
    assert run._metric_values_with_warning_audit(None, None) == (  # noqa: SLF001
        sentinel,
        "a" * 64,
    )

    def unexpected(_render: object, _target: object) -> tuple[object, str]:
        warnings.warn("foreign warning", UserWarning, stacklevel=1)
        return sentinel, "a" * 64

    monkeypatch.setattr(run.quality, "exact_metric_values", unexpected)
    with pytest.raises(run.LifecycleError, match="warning surface"):
        run._metric_values_with_warning_audit(None, None)  # noqa: SLF001


def _journal(tmp_path: Path) -> tuple[run.EventJournal, run.AccessGuard]:
    journal = run.EventJournal(tmp_path / "artifact/journal/events")
    return journal, run.AccessGuard(journal, artifact_root=tmp_path / "artifact")


def test_source_manifest_is_exact_stable_and_complete() -> None:
    first = run.source_manifest()
    second = run.source_manifest()
    assert first == second
    assert first["task_sha256"] == run.TASK_SHA256
    paths = {record["path"] for record in first["files"]}
    required = {
        "pyproject.toml",
        "tasks/COMP-008-mean-conditioned-entropy-oracle.md",
        "tasks/COMP-009-ssp2e-actual-coder.md",
        "tasks/COMP-010-ssp2e-captured-replay-repair.md",
        "tasks/COMP-011-complete-stream-rgb-vq.md",
        "benchmarks/assets/ssp2e_cdf_v1.bin",
        "benchmarks/assets/ssp2e_cost_q24_v1.bin",
        "benchmarks/ssp2e_range_ext.cpp",
        "benchmarks/ssp2v_lloyd_ext.cpp",
        "benchmarks/ssp2v_actual_run.py",
        "tests/test_ssp2v_actual_run.py",
        "tests/test_ssp2v_actual_run_fresh_broker.py",
        "src/structsplat/cuda/render_ext.cpp",
        "src/structsplat/cuda/render_ext.cu",
    }
    assert required <= paths
    assert any(path.startswith("src/structsplat/") for path in paths)
    assert {record["closure"] for record in first["files"]} <= {
        "explicit",
        "transitive-or-production",
    }


def test_source_manifest_dependency_discovery_and_hashing_are_guarded(tmp_path: Path) -> None:
    journal, guard = _journal(tmp_path)
    manifest = run.source_manifest(guard=guard)
    events = run.EventJournal.verify(journal.root)
    assert manifest["source_manifest_sha256"]
    assert events
    assert all(event["path_class"] == "repository_source" for event in events)
    assert any(event["purpose"] == "discover transitive local source imports" for event in events)
    assert any(event["purpose"] == "hash source closure" for event in events)


def test_exclusive_write_is_create_only_and_readonly(tmp_path: Path) -> None:
    path = tmp_path / "immutable.bin"
    run._exclusive_write(path, b"one")
    assert path.read_bytes() == b"one"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(run.LifecycleError, match="replace"):
        run._exclusive_write(path, b"two")
    assert path.read_bytes() == b"one"


def test_exclusive_write_recovers_writable_partial_pending(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    payload = b"complete immutable artifact"
    pending = run._pending_publication_path(path, payload)
    pending.write_bytes(payload[:7])

    run._exclusive_write(path, payload)

    assert path.read_bytes() == payload
    assert path.stat().st_mode & 0o222 == 0
    assert not pending.exists()


@pytest.mark.parametrize("already_linked", [False, True])
def test_exclusive_write_finishes_complete_pending_publication(
    tmp_path: Path, already_linked: bool
) -> None:
    path = tmp_path / "artifact.bin"
    payload = b"complete immutable artifact"
    pending = run._pending_publication_path(path, payload)
    pending.write_bytes(payload)
    pending.chmod(0o444)
    if already_linked:
        os.link(pending, path)

    run._exclusive_write(path, payload)

    assert path.read_bytes() == payload
    assert path.stat().st_mode & 0o222 == 0
    assert path.stat().st_nlink == 1
    assert not pending.exists()


def test_exclusive_write_and_journal_reject_symlink_parents(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(run.LifecycleError, match="symlink"):
        run._exclusive_write(linked / "artifact.bin", b"x")
    with pytest.raises(run.LifecycleError, match="symlink"):
        run.EventJournal(linked / "journal/events")
    assert not (real / "artifact.bin").exists()


def test_journal_resume_preserves_prefix_byte_for_byte(tmp_path: Path) -> None:
    root = tmp_path / "journal/events"
    first = run.EventJournal(root)
    first.append(
        stage="preflight",
        operation="open-read-complete",
        path_class="repository_source",
        purpose="first",
        path="repository:first",
    )
    prefix = (root / "00000000.json").read_bytes()
    resumed = run.EventJournal(root, resume=True)
    resumed.append(
        stage="preflight",
        operation="open-read-complete",
        path_class="upstream_metadata",
        purpose="second",
        path="repository:second",
    )
    assert (root / "00000000.json").read_bytes() == prefix
    records = run.EventJournal.verify(root)
    assert [record["sequence"] for record in records] == [0, 1]
    assert records[1]["previous_event_sha256"] == records[0]["event_sha256"]


def test_journal_refuses_implicit_resume(tmp_path: Path) -> None:
    root = tmp_path / "journal/events"
    run.EventJournal(root)
    with pytest.raises(run.LifecycleError, match="explicit resume"):
        run.EventJournal(root)


def test_sealed_journal_prefix_allows_later_append_but_not_wrong_head(tmp_path: Path) -> None:
    root = tmp_path / "journal/events"
    journal = run.EventJournal(root)
    first = journal.append(
        stage="preflight",
        operation="authorize-create-immutable",
        path_class="artifact_runtime",
        purpose="preflight seal",
        path="artifact:preflight.json",
    )
    binding = {"length": 1, "head_sha256": first["event_sha256"]}
    journal.bind_stage_seal("preflight", "a" * 64)
    journal.append(
        stage="import-streams",
        operation="create-immutable-complete",
        path_class="artifact_stream",
        purpose="later stage",
        path="artifact:stream.bin",
        prior_stage_seal="a" * 64,
    )
    events = run.EventJournal.verify(root)
    run.verify_journal_prefix(events, binding)
    with pytest.raises(run.LifecycleError, match="prefix"):
        run.verify_journal_prefix(events, {"length": 1, "head_sha256": "0" * 64})
    with pytest.raises(run.LifecycleError, match="prefix"):
        run.verify_journal_prefix(events, {"length": 3, "head_sha256": first["event_sha256"]})


@pytest.mark.parametrize("hostility", ["gap", "extra", "writable", "corrupt"])
def test_journal_rejects_hostile_resume_state(tmp_path: Path, hostility: str) -> None:
    root = tmp_path / "journal/events"
    journal = run.EventJournal(root)
    for index in range(2):
        journal.append(
            stage="preflight",
            operation="open-read-complete",
            path_class="repository_source",
            purpose=str(index),
            path=f"repository:{index}",
        )
    if hostility == "gap":
        (root / "00000001.json").rename(root / "00000002.json")
    elif hostility == "extra":
        (root / "foreign").write_bytes(b"x")
    elif hostility == "writable":
        os.chmod(root / "00000000.json", 0o644)
    else:
        path = root / "00000000.json"
        os.chmod(path, 0o644)
        path.write_bytes(path.read_bytes().replace(b'"purpose":"0"', b'"purpose":"x"'))
        os.chmod(path, 0o444)
    with pytest.raises(run.LifecycleError):
        run.EventJournal(root, resume=True)


def test_guard_requires_exact_registration_and_stage(tmp_path: Path) -> None:
    journal, guard = _journal(tmp_path)
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    with pytest.raises(run.LifecycleError, match="not explicitly registered"):
        guard.read_bytes(source, stage="preflight", purpose="unregistered")
    guard.register(source, "development_stream")
    guard.bind_stage_seal("preflight", "a" * 64)
    with pytest.raises(run.LifecycleError, match="forbidden"):
        guard.read_bytes(source, stage="preflight", purpose="too early")
    assert guard.read_bytes(
        source,
        stage="import-streams",
        purpose="authorized",
        prior_stage_seal="a" * 64,
    ) == b"source"
    assert run.EventJournal.verify(journal.root)[-1]["path_class"] == "development_stream"


def test_guard_rejects_target_before_complete_candidate_seal(tmp_path: Path) -> None:
    _, guard = _journal(tmp_path)
    target = tmp_path / "target.png"
    target.write_bytes(b"not-decoded")
    guard.register(target, "upstream_target")
    guard.bind_stage_seal("fit-encode-cold-decode-dev", "b" * 64)
    guard.bind_candidate_set_seal("c" * 64)
    with pytest.raises(run.LifecycleError, match="candidate-set seal"):
        guard.read_bytes(
            target,
            stage="import-targets",
            purpose="too early",
            prior_stage_seal="b" * 64,
        )
    assert guard.read_bytes(
        target,
        stage="import-targets",
        purpose="copy sealed target",
        candidate_seal="c" * 64,
        prior_stage_seal="b" * 64,
    ) == b"not-decoded"


def test_guard_requires_both_seals_for_artifact_target(tmp_path: Path) -> None:
    _, guard = _journal(tmp_path)
    target = tmp_path / "artifact-target.png"
    target.write_bytes(b"copy")
    guard.register(target, "artifact_target")
    guard.bind_stage_seal("import-targets", "d" * 64)
    guard.bind_candidate_set_seal("c" * 64)
    guard.bind_target_copy_seal("e" * 64)
    for kwargs in ({}, {"candidate_seal": "c" * 64}):
        with pytest.raises(run.LifecycleError):
            guard.read_bytes(
                target,
                stage="quality-select-dev",
                purpose="score",
                prior_stage_seal="d" * 64,
                **kwargs,
            )
    assert guard.read_bytes(
        target,
        stage="quality-select-dev",
        purpose="score",
        candidate_seal="c" * 64,
        target_seal="e" * 64,
        prior_stage_seal="d" * 64,
    ) == b"copy"


def test_guard_rejects_unverified_or_wrong_stage_and_authority_seals(tmp_path: Path) -> None:
    _, guard = _journal(tmp_path)
    stream = tmp_path / "stream.sspl1"
    stream.write_bytes(b"x")
    guard.register(stream, "development_stream")
    with pytest.raises(run.LifecycleError, match="exact verified preflight seal"):
        guard.read_bytes(
            stream,
            stage="import-streams",
            purpose="missing prior",
            prior_stage_seal="a" * 64,
        )
    guard.bind_stage_seal("preflight", "a" * 64)
    with pytest.raises(run.LifecycleError, match="exact verified preflight seal"):
        guard.read_bytes(
            stream,
            stage="import-streams",
            purpose="wrong prior",
            prior_stage_seal="b" * 64,
        )
    with pytest.raises(run.LifecycleError, match="SHA-256"):
        guard.bind_candidate_set_seal("not-a-digest")
    guard.bind_candidate_set_seal("c" * 64)
    with pytest.raises(run.LifecycleError, match="rebound"):
        guard.bind_candidate_set_seal("d" * 64)


def test_confirmation_is_inaccessible_in_every_stage(tmp_path: Path) -> None:
    _, guard = _journal(tmp_path)
    confirmation = tmp_path / "confirmation.bin"
    confirmation.write_bytes(b"forbidden")
    guard.register(confirmation, "confirmation")
    for stage in run.STAGES:
        with pytest.raises(run.LifecycleError):
            guard.read_bytes(confirmation, stage=stage, purpose="hostile probe")


def test_guard_rejects_final_symlink_and_symlink_parent(tmp_path: Path) -> None:
    _, guard = _journal(tmp_path)
    real = tmp_path / "real.bin"
    real.write_bytes(b"x")
    final_link = tmp_path / "link.bin"
    final_link.symlink_to(real)
    guard.register(final_link, "upstream_metadata")
    with pytest.raises(run.LifecycleError, match="symlink"):
        guard.read_bytes(final_link, stage="preflight", purpose="hostile link")

    real_dir = tmp_path / "real-dir"
    real_dir.mkdir()
    (real_dir / "metadata.json").write_text("{}", encoding="utf-8")
    parent_link = tmp_path / "linked-dir"
    parent_link.symlink_to(real_dir, target_is_directory=True)
    linked_child = parent_link / "metadata.json"
    guard.register(linked_child, "upstream_metadata")
    with pytest.raises(run.LifecycleError, match="symlink"):
        guard.read_bytes(linked_child, stage="preflight", purpose="hostile parent")


def _small_manifest() -> dict[str, object]:
    files = []
    for relative in ("benchmarks/__init__.py", "pyproject.toml"):
        payload = (run.ROOT / relative).read_bytes()
        files.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "closure": "explicit",
            }
        )
    return actual.seal_record(
        {
            "schema": run.SOURCE_SCHEMA,
            "task_sha256": run.TASK_SHA256,
            "files": files,
        },
        "source_manifest_sha256",
    )


def test_source_archive_is_deterministic_safe_and_relocatable(tmp_path: Path) -> None:
    manifest = _small_manifest()
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    run.write_source_archive(first, manifest)
    run.write_source_archive(second, manifest)
    assert first.read_bytes() == second.read_bytes()
    extracted = tmp_path / "randomized-root-8d51/source"
    observed = run.safe_extract_source_archive(first, extracted, expected_manifest=manifest)
    assert observed == manifest
    assert (extracted / "pyproject.toml").read_bytes() == (run.ROOT / "pyproject.toml").read_bytes()
    assert {path.relative_to(extracted).as_posix() for path in extracted.rglob("*") if path.is_file()} == {
        "SOURCE_MANIFEST.json",
        "benchmarks/__init__.py",
        "pyproject.toml",
    }


def _hostile_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, "w") as archive:
        for info, payload in members:
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


@pytest.mark.parametrize(
    "hostility", ["traversal", "absolute", "symlink", "hardlink", "duplicate"]
)
def test_source_archive_rejects_hostile_members(tmp_path: Path, hostility: str) -> None:
    path = tmp_path / f"{hostility}.tar"
    if hostility == "traversal":
        members = [(tarfile.TarInfo("../escape"), b"x")]
    elif hostility == "absolute":
        members = [(tarfile.TarInfo("/escape"), b"x")]
    elif hostility == "symlink":
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "outside"
        members = [(info, b"")]
    elif hostility == "hardlink":
        info = tarfile.TarInfo("link")
        info.type = tarfile.LNKTYPE
        info.linkname = "outside"
        members = [(info, b"")]
    else:
        members = [(tarfile.TarInfo("same"), b"a"), (tarfile.TarInfo("same"), b"b")]
    _hostile_tar(path, members)
    with pytest.raises(run.LifecycleError):
        run.safe_extract_source_archive(path, tmp_path / "extract")
    assert not (tmp_path / "escape").exists()


def test_upstream_binding_opens_only_sealed_metadata_and_captured_source(tmp_path: Path) -> None:
    journal, guard = _journal(tmp_path)
    record = run.bind_upstream_metadata(
        guard,
        run.DEFAULT_COMP008,
        run.DEFAULT_COMP009,
        run.DEFAULT_COMP010,
        artifact_root=tmp_path / "artifact",
    )
    actual.verify_record(record, "authority_manifest_sha256")
    assert len(record["comp008"]["cells"]) == 16
    assert len(record["comp008"]["targets"]) == 8
    assert len(record["comp009"]["cells"]) == 16
    assert record["comp010"]["captured_child_count"] == 2
    assert record["development_streams_opened"] is False
    assert record["target_or_pixel_payloads_opened"] is False
    assert record["confirmation_payloads_accessed"] is False
    events = run.EventJournal.verify(journal.root)
    opened = [event for event in events if event["operation"] == "open-read-complete"]
    assert {event["path_class"] for event in opened} == {
        "upstream_metadata",
        "runtime_dependency",
        "artifact_runtime",
    }
    assert all(".sspl1" not in event["path"] for event in opened)
    assert all(".png" not in event["path"] for event in opened)
    assert all(".npz" not in event["path"] for event in opened)


def test_preflight_rejects_nonempty_directory_before_environment_or_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "occupied"
    outdir.mkdir()
    (outdir / "foreign").write_bytes(b"x")
    monkeypatch.setattr(
        run.quality,
        "frozen_runtime_environment",
        lambda: pytest.fail("environment must not be inspected after nonempty-dir rejection"),
    )
    with pytest.raises(run.LifecycleError, match="truly empty"):
        run.preflight(outdir, tmp_path / "absent8", tmp_path / "absent9", tmp_path / "absent10")


def test_preflight_rejects_symlink_output_root_before_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(
        run.quality,
        "frozen_runtime_environment",
        lambda: pytest.fail("environment must not be inspected through a symlink root"),
    )
    with pytest.raises(run.LifecycleError, match="symlink"):
        run.preflight(linked)


def test_public_later_stages_require_a_sealed_predecessor_without_touching_outdir(
    tmp_path: Path,
) -> None:
    absent = tmp_path / "must-remain-absent"
    for stage in run.STAGES[1:]:
        with pytest.raises(run.LifecycleError, match="preflight"):
            run.main([stage, "--outdir", str(absent)])
        assert not absent.exists()


def test_parser_exposes_exact_eight_stage_lifecycle() -> None:
    parser = run.build_parser()
    for stage in run.STAGES:
        args = parser.parse_args([stage, "--outdir", "/tmp/comp011-test"])
        assert args.command == stage


def test_worker_environment_is_offline_and_prefers_artifact_metrics(tmp_path: Path) -> None:
    environment = run._worker_environment(tmp_path)
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(
        tmp_path / "runtime/metrics/site"
    )
    assert environment["TORCH_HOME"] == str(tmp_path / "runtime/metrics/torch")
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["LD_PRELOAD"] == run.quality.REQUIRED_RENDERER_PRELOAD


_TEST_LSCPU_EXECUTABLE = {
    "path": "/usr/bin/lscpu",
    "bytes": 123,
    "sha256": "a" * 64,
    "version": {
        "command": ["/usr/bin/lscpu", "--version"],
        "returncode": 0,
        "stdout": "lscpu test version\n",
        "stderr": "",
        "output_sha256": hashlib.sha256(b"lscpu test version\n").hexdigest(),
    },
}

_TEST_NVIDIA_SMI_EXECUTABLE = {
    "path": "/usr/bin/nvidia-smi",
    "bytes": 456,
    "sha256": "b" * 64,
}


def _lscpu_command_record(entries: list[dict[str, object]]) -> dict[str, object]:
    stdout = __import__("json").dumps({"lscpu": entries})
    return {
        "command": [_TEST_LSCPU_EXECUTABLE["path"], "--json"],
        "returncode": 0,
        "stdout": stdout,
        "stderr": "",
        "output_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
    }


def _nvidia_smi_command_record(
    stdout: str = (
        "0, NVIDIA RTX TEST, GPU-0000, 999.1, 24564, 8.9\n"
    ),
) -> dict[str, object]:
    command = [
        _TEST_NVIDIA_SMI_EXECUTABLE["path"],
        f"--query-gpu={','.join(run._NVIDIA_SMI_QUERY_FIELDS)}",  # noqa: SLF001
        "--format=csv,noheader,nounits",
    ]
    return {
        "command": command,
        "returncode": 0,
        "stdout": stdout,
        "stderr": "",
        "output_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
    }


def test_cpuinfo_identity_is_repeat_stable_and_excludes_only_live_clock() -> None:
    first = b"""processor : 1
vendor_id : GenuineIntel
physical id : 0
core id : 1
model name : Example CPU
microcode : 0x65
cpu MHz : 4901.125
flags : fpu avx2

processor : 0
vendor_id : GenuineIntel
physical id : 0
core id : 0
model name : Example CPU
microcode : 0x65
cpu MHz : 812.250
flags : fpu avx2
"""
    second = first.replace(b"4901.125", b"800.000").replace(b"812.250", b"5300.000")

    identity = run._cpuinfo_identity_record(first)
    assert identity == run._cpuinfo_identity_record(first)
    assert identity == run._cpuinfo_identity_record(second)
    assert identity["excluded_volatile_fields"] == ["cpu MHz"]
    assert [entry["processor"] for entry in identity["logical_processors"]] == ["0", "1"]
    assert all("cpu MHz" not in entry for entry in identity["logical_processors"])


@pytest.mark.parametrize(
    "mutation",
    [
        b"microcode : 0x66\n",
        b"physical id : 1\n",
        b"new stable capability : enabled\n",
    ],
)
def test_cpuinfo_identity_rejects_meaningful_stable_or_extra_field_drift(
    mutation: bytes,
) -> None:
    baseline = b"""processor : 0
vendor_id : GenuineIntel
physical id : 0
model name : Example CPU
microcode : 0x65
cpu MHz : 1200.000
flags : fpu avx2
"""
    key = mutation.split(b":", 1)[0].strip()
    retained = b"\n".join(
        line for line in baseline.splitlines() if line.split(b":", 1)[0].strip() != key
    ) + b"\n"
    changed = retained + mutation
    assert (
        run._cpuinfo_identity_record(changed)["identity_sha256"]
        != run._cpuinfo_identity_record(baseline)["identity_sha256"]
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"processor : 0\ncpu MHz : 1\nmalformed\n",
        b"processor : 0\ncpu MHz : 1\nmodel name : one\nmodel name : two\n",
        b"processor : 0\ncpu MHz : 1\ncpu MHz : 2\n",
        b"processor : 0\ncpu MHz : 1\n\nprocessor : 0\ncpu MHz : 2\n",
        b"processor : 0\ncpu mhz : 1\n",
    ],
)
def test_cpuinfo_identity_rejects_malformed_and_duplicate_fields(payload: bytes) -> None:
    with pytest.raises(run.LifecycleError, match="malformed|duplicate|absent"):
        run._cpuinfo_identity_record(payload)


def test_lscpu_identity_is_repeat_stable_and_excludes_only_scaling_telemetry() -> None:
    stable = [
        {"field": "Architecture:", "data": "x86_64"},
        {"field": "CPU(s):", "data": "16"},
        {"field": "Thread(s) per core:", "data": "2"},
        {"field": "Core(s) per socket:", "data": "8"},
        {"field": "Socket(s):", "data": "1"},
        {"field": "CPU max MHz:", "data": "5300.0000"},
        {"field": "CPU min MHz:", "data": "800.0000"},
        {"field": "Vulnerability Spectre v2:", "data": "Mitigation; Enhanced IBRS"},
    ]
    first = _lscpu_command_record(
        [stable[2], {"field": "CPU(s) scaling MHz:", "data": "34%"}, *stable[:2], *stable[3:]]
    )
    second = _lscpu_command_record(
        [*reversed(stable), {"field": "CPU(s) scaling MHz:", "data": "97%"}]
    )

    identity = run._lscpu_identity_record(first, executable=_TEST_LSCPU_EXECUTABLE)
    assert identity == run._lscpu_identity_record(
        first, executable=_TEST_LSCPU_EXECUTABLE
    )
    assert identity == run._lscpu_identity_record(
        second, executable=_TEST_LSCPU_EXECUTABLE
    )
    assert identity["excluded_volatile_fields"] == ["CPU(s) scaling MHz:"]
    assert "CPU(s) scaling MHz:" not in identity["fields"]
    assert identity["fields"]["CPU max MHz:"] == "5300.0000"
    assert identity["fields"]["CPU min MHz:"] == "800.0000"
    assert identity["fields"]["Vulnerability Spectre v2:"] == "Mitigation; Enhanced IBRS"


@pytest.mark.parametrize(
    "mutation",
    [
        {"field": "CPU max MHz:", "data": "5100.0000"},
        {"field": "Vulnerability Spectre v2:", "data": "Vulnerable"},
        {"field": "New stable topology field:", "data": "one"},
    ],
)
def test_lscpu_identity_rejects_meaningful_stable_or_extra_field_drift(
    mutation: dict[str, object],
) -> None:
    baseline_entries = [
        {"field": "Architecture:", "data": "x86_64"},
        {"field": "CPU(s) scaling MHz:", "data": "55%"},
        {"field": "CPU max MHz:", "data": "5300.0000"},
        {"field": "Vulnerability Spectre v2:", "data": "Mitigation"},
    ]
    changed_entries = [entry for entry in baseline_entries if entry["field"] != mutation["field"]]
    changed_entries.append(mutation)
    baseline = run._lscpu_identity_record(
        _lscpu_command_record(baseline_entries), executable=_TEST_LSCPU_EXECUTABLE
    )
    changed = run._lscpu_identity_record(
        _lscpu_command_record(changed_entries), executable=_TEST_LSCPU_EXECUTABLE
    )
    assert changed["identity_sha256"] != baseline["identity_sha256"]


@pytest.mark.parametrize(
    "executable_mutation",
    [
        {"sha256": "b" * 64},
        {
            "version": {
                "command": ["/usr/bin/lscpu", "--version"],
                "returncode": 0,
                "stdout": "lscpu changed version\n",
                "stderr": "",
                "output_sha256": hashlib.sha256(b"lscpu changed version\n").hexdigest(),
            }
        },
    ],
)
def test_lscpu_identity_binds_executable_and_version_identity(
    executable_mutation: dict[str, object],
) -> None:
    entries = [
        {"field": "Architecture:", "data": "x86_64"},
        {"field": "CPU(s) scaling MHz:", "data": "55%"},
    ]
    baseline = run._lscpu_identity_record(
        _lscpu_command_record(entries), executable=_TEST_LSCPU_EXECUTABLE
    )
    mutated_executable = {**_TEST_LSCPU_EXECUTABLE, **executable_mutation}
    mutated_command = _lscpu_command_record(entries)
    mutated_command["command"] = [mutated_executable["path"], "--json"]
    mutated = run._lscpu_identity_record(
        mutated_command, executable=mutated_executable
    )
    assert mutated["identity_sha256"] != baseline["identity_sha256"]


@pytest.mark.parametrize(
    "entries",
    [
        [{"field": "Architecture:", "data": "x86_64", "extra": "bad"}],
        [
            {"field": "Architecture:", "data": "x86_64"},
            {"field": "Architecture:", "data": "amd64"},
        ],
        [
            {"field": "CPU(s) scaling MHz:", "data": "20%"},
            {"field": "CPU(s) scaling MHz:", "data": "90%"},
        ],
        [
            {"field": "Architecture:", "data": "x86_64"},
            {"field": "CPU(s) scaling MHz :", "data": "90%"},
        ],
    ],
)
def test_lscpu_identity_rejects_malformed_and_duplicate_fields(
    entries: list[dict[str, object]],
) -> None:
    with pytest.raises(run.LifecycleError, match="malformed|duplicate|absent"):
        run._lscpu_identity_record(
            _lscpu_command_record(entries), executable=_TEST_LSCPU_EXECUTABLE
        )


def test_bound_lscpu_identity_requires_exact_seal_and_live_executable() -> None:
    identity = run._lscpu_identity_record(
        _lscpu_command_record(
            [
                {"field": "Architecture:", "data": "x86_64"},
                {"field": "CPU(s):", "data": "16"},
                {"field": "CPU(s) scaling MHz:", "data": "55%"},
            ]
        ),
        executable=_TEST_LSCPU_EXECUTABLE,
    )
    assert (
        run._bound_lscpu_identity_record(
            identity, executable=_TEST_LSCPU_EXECUTABLE
        )
        == identity
    )

    damaged = {**identity, "stderr": "changed"}
    with pytest.raises(run.LifecycleError, match="seal mismatch"):
        run._bound_lscpu_identity_record(
            damaged, executable=_TEST_LSCPU_EXECUTABLE
        )

    body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    body["executable"] = {**_TEST_LSCPU_EXECUTABLE, "sha256": "e" * 64}
    resealed = {
        **body,
        "identity_sha256": run._sha256_bytes(run._canonical_json(body)),
    }
    with pytest.raises(run.LifecycleError, match="live executable"):
        run._bound_lscpu_identity_record(
            resealed, executable=_TEST_LSCPU_EXECUTABLE
        )


def test_bound_nvidia_smi_identity_requires_exact_seal_and_live_executable() -> None:
    identity = run._nvidia_smi_identity_record(  # noqa: SLF001
        _nvidia_smi_command_record(),
        executable=_TEST_NVIDIA_SMI_EXECUTABLE,
    )
    assert identity["rows"] == [
        {
            "index": "0",
            "name": "NVIDIA RTX TEST",
            "uuid": "GPU-0000",
            "driver_version": "999.1",
            "memory.total": "24564",
            "compute_cap": "8.9",
        }
    ]
    assert (
        run._bound_nvidia_smi_identity_record(  # noqa: SLF001
            identity,
            executable=_TEST_NVIDIA_SMI_EXECUTABLE,
        )
        == identity
    )

    damaged = {**identity, "stderr": "changed"}
    with pytest.raises(run.LifecycleError, match="seal mismatch"):
        run._bound_nvidia_smi_identity_record(  # noqa: SLF001
            damaged,
            executable=_TEST_NVIDIA_SMI_EXECUTABLE,
        )

    changed_executable = {
        **_TEST_NVIDIA_SMI_EXECUTABLE,
        "sha256": "c" * 64,
    }
    with pytest.raises(run.LifecycleError, match="live executable"):
        run._bound_nvidia_smi_identity_record(  # noqa: SLF001
            identity,
            executable=changed_executable,
        )

    body = {
        key: value
        for key, value in identity.items()
        if key != "identity_sha256"
    }
    body["unexpected"] = True
    resealed = {
        **body,
        "identity_sha256": run._sha256_bytes(  # noqa: SLF001
            run._canonical_json(body)  # noqa: SLF001
        ),
    }
    with pytest.raises(run.LifecycleError, match="sealed inventory"):
        run._bound_nvidia_smi_identity_record(  # noqa: SLF001
            resealed,
            executable=_TEST_NVIDIA_SMI_EXECUTABLE,
        )


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "0, too, few\n",
        "x, NVIDIA RTX TEST, GPU-0000, 999.1, 24564, 8.9\n",
        (
            "0, NVIDIA RTX TEST, GPU-0000, 999.1, 24564, 8.9\n"
            "0, NVIDIA RTX TEST 2, GPU-0001, 999.1, 24564, 8.9\n"
        ),
        (
            "0, NVIDIA RTX TEST, GPU-0000, 999.1, 24564, 8.9\n"
            "1, NVIDIA RTX TEST 2, GPU-0000, 999.1, 24564, 8.9\n"
        ),
    ],
)
def test_nvidia_smi_identity_rejects_malformed_inventory(stdout: str) -> None:
    with pytest.raises(run.LifecycleError, match="malformed"):
        run._nvidia_smi_identity_record(  # noqa: SLF001
            _nvidia_smi_command_record(stdout),
            executable=_TEST_NVIDIA_SMI_EXECUTABLE,
        )


def test_live_environment_binding_is_immediately_repeatable() -> None:
    assert run.environment_record() == run.environment_record()


def test_environment_binding_rejects_one_sided_bound_hardware() -> None:
    with pytest.raises(run.LifecycleError, match="provide both"):
        run.environment_record(bound_lscpu={})
    with pytest.raises(run.LifecycleError, match="provide both"):
        run.environment_record(bound_nvidia_smi={})


def test_bound_hardware_environment_replays_without_gpu_device_authority() -> None:
    outer = run.environment_record()
    source_root = Path(run.__file__).resolve(strict=True).parents[1]
    script = r"""
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
sys.path[:0] = [str(root), str(root / "src")]
from benchmarks import ssp2v_actual_run as runner

bound = json.loads(bytes.fromhex(sys.argv[2]).decode("utf-8"))
original_command_record = runner._command_record
def guarded_command_record(command, *, timeout=60.0):
    if pathlib.Path(command[0]).name == "nvidia-smi":
        raise RuntimeError("captured environment attempted live nvidia-smi")
    return original_command_record(command, timeout=timeout)
runner._command_record = guarded_command_record
record = runner.environment_record(
    bound_lscpu=bound["lscpu"],
    bound_nvidia_smi=bound["nvidia_smi"],
)
sys.stdout.buffer.write(runner._canonical_json(record))
sys.stdout.buffer.flush()
"""
    command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        script,
        os.fspath(source_root),
        run._canonical_json(outer).hex(),
    ]
    replayed = run._run_json_worker(
        command,
        outdir=None,
        timeout=60,
        read_only=[source_root],
        read_write=run._sandbox_process_read_write(),
        cwd=source_root,
        environment=run._worker_environment(None, source_root=source_root),
    )
    assert replayed == outer


def test_end_preflight_environment_recheck_fails_closed_on_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {"hardware": "captured"}
    monkeypatch.setattr(run, "environment_record", lambda: {"hardware": "captured"})
    run._verify_preflight_environment_unchanged(captured)
    monkeypatch.setattr(run, "environment_record", lambda: {"hardware": "changed"})
    with pytest.raises(run.LifecycleError, match="environment changed during preflight"):
        run._verify_preflight_environment_unchanged(captured)


def test_renderer_source_record_binds_production_adapter_and_cuda_sources() -> None:
    records = run._renderer_source_record()
    paths = {record["path"] for record in records}
    assert paths == {
        "src/structsplat/cuda_render.py",
        "src/structsplat/render.py",
        "src/structsplat/gaussians.py",
        "src/structsplat/cuda/render_ext.cpp",
        "src/structsplat/cuda/render_ext.cu",
    }
    assert all(record["bytes"] > 0 and len(record["sha256"]) == 64 for record in records)


@pytest.mark.parametrize(
    "value",
    [".", "..", "../x", "a/../b", "/absolute", "a//b", "a\\b", "", "a/./b"],
)
def test_artifact_relative_paths_reject_aliases_and_traversal(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(run.LifecycleError):
        run._safe_relative_path(value)
    with pytest.raises(run.LifecycleError):
        run._artifact_path(tmp_path, value)


def test_artifact_relative_path_accepts_one_canonical_nested_path(tmp_path: Path) -> None:
    assert run._safe_relative_path("one/two.bin") == Path("one/two.bin")
    assert run._artifact_path(tmp_path, "one/two.bin") == tmp_path / "one/two.bin"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bytes", True),
        ("bytes", 1.0),
        ("bytes", -1),
        ("sha256", "not-a-digest"),
        ("sha256", "A" * 64),
    ],
)
def test_regular_record_rejects_malformed_size_and_digest(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"x")
    record: dict[str, object] = {
        "bytes": 1,
        "sha256": hashlib.sha256(b"x").hexdigest(),
    }
    record[field] = value
    with pytest.raises(run.LifecycleError):
        run._verify_regular_record(path, record, readonly=False)


@pytest.mark.parametrize(
    ("operation", "path_class"),
    [
        ("authorize-create-immutable", "repository_source"),
        ("create-immutable-complete", "development_stream"),
        ("resume-verify-immutable-complete", "upstream_metadata"),
        ("commit-immutable-cell-directory", "artifact_runtime"),
    ],
)
def test_journal_rejects_illegal_operation_path_class_pairs(
    tmp_path: Path, operation: str, path_class: str
) -> None:
    journal = run.EventJournal(tmp_path / "journal/events")
    with pytest.raises(run.LifecycleError):
        journal.append(
            stage="preflight",
            operation=operation,
            path_class=path_class,
            purpose="hostile pair",
            path="repository:hostile",
        )


def test_stage_journal_segment_requires_immediate_exact_completion(tmp_path: Path) -> None:
    journal = run.EventJournal(tmp_path / "journal/events")
    authorization = journal.append(
        stage="preflight",
        operation="authorize-create-immutable",
        path_class="artifact_runtime",
        purpose="authorize",
        path="artifact:preflight.json",
    )
    binding = {"length": 1, "head_sha256": authorization["event_sha256"]}
    payload = b"sealed-manifest\n"
    journal.append(
        stage="preflight",
        operation="create-immutable-complete",
        path_class="artifact_runtime",
        purpose="complete",
        path="artifact:preflight.json",
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        artifact_bytes=len(payload),
    )
    events = run.EventJournal.verify(journal.root)
    run._verify_stage_journal_segment(
        events,
        binding,
        stage="preflight",
        manifest_path="preflight.json",
        manifest_payload=payload,
        prior_stage_seal=None,
    )
    with pytest.raises(run.LifecycleError, match="completion"):
        run._verify_stage_journal_segment(
            events[:1],
            binding,
            stage="preflight",
            manifest_path="preflight.json",
            manifest_payload=payload,
            prior_stage_seal=None,
        )
    hostile = [dict(item) for item in events]
    hostile[1]["artifact_bytes"] += 1
    with pytest.raises(run.LifecycleError, match="completion"):
        run._verify_stage_journal_segment(
            hostile,
            binding,
            stage="preflight",
            manifest_path="preflight.json",
            manifest_payload=payload,
            prior_stage_seal=None,
        )


def test_transitive_source_closure_follows_relative_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "benchmarks"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "a.py").write_text("from . import b\n", encoding="utf-8")
    (package / "b.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    assert run._transitive_python_sources(["benchmarks/a.py"]) == {
        "benchmarks/__init__.py",
        "benchmarks/a.py",
        "benchmarks/b.py",
    }


def test_json_worker_attests_child_relative_proc_task_profile() -> None:
    command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        "import os; os.write(1, b'{\\\"ok\\\":true}\\n')",
    ]
    assert run._run_json_worker(
        command,
        outdir=None,
        timeout=30,
        read_write=run._sandbox_process_read_write(),
        include_repository_source=False,
    ) == {"ok": True}


def test_measurement_sessions_validate_arms_and_separate_replay_namespace() -> None:
    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    quality_seal = "a" * 64
    ordinary = run._measurement_session_sha256(
        kind="primary",
        image_id=image_id,
        bits=bits,
        quality_stage_sha256=quality_seal,
        candidate=actual.VARIANT_ORDER[0],
        q_name=actual.PRIMARY_ORDER[0],
    )
    replay = run._measurement_session_sha256(
        kind="primary",
        image_id=image_id,
        bits=bits,
        quality_stage_sha256=quality_seal,
        candidate=actual.VARIANT_ORDER[0],
        q_name=actual.PRIMARY_ORDER[0],
        replay_phase_sha256="b" * 64,
    )
    assert ordinary != replay
    with pytest.raises(run.LifecycleError):
        run._measurement_session_sha256(
            kind="primary",
            image_id=image_id,
            bits=bits,
            quality_stage_sha256=quality_seal,
        )
    with pytest.raises(run.LifecycleError):
        run._measurement_session_sha256(
            kind="secondary",
            image_id=image_id,
            bits=bits,
            quality_stage_sha256=quality_seal,
            candidate=actual.VARIANT_ORDER[0],
        )


@pytest.mark.parametrize("state", ["invalid", "method_failure"])
def test_unselected_replay_quality_uses_all_eight_variants(state: str) -> None:
    assert run._decision_relevant_variant_prefix(
        {"selection": {"state": state, "ordered_labels": []}}
    ) == list(actual.VARIANT_ORDER)


def test_replay_phase_b_rejects_empty_target_derived_grids(tmp_path: Path) -> None:
    candidate_replay = {"state": "complete_candidate_set", "cells": []}
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
    phase_b = actual.seal_record(
        {
            "schema": run.REPLAY_PHASE_B_SCHEMA,
            "protocol": run.PROTOCOL,
            "task_sha256": run.TASK_SHA256,
            "source_manifest_sha256": "1" * 64,
            "candidate_replay": candidate_replay,
            "replay_phase_a_sha256": phase_a["replay_phase_a_sha256"],
            "quality_replay": [],
            "resource_replay": [],
            "analysis_gate_signature": run._analysis_gate_signature({}),
            "analysis_record_sha256": "2" * 64,
            "benchmark_stage_sha256": "3" * 64,
            "targets_opened": True,
            "confirmation_payloads_accessed": False,
        },
        "captured_replay_worker_sha256",
    )
    with pytest.raises(run.LifecycleError, match="incomplete"):
        run._verify_replay_phase_b(
            tmp_path,
            phase_b,
            phase_a,
            {"state": "complete_candidate_set", "cells": []},
            {"state": "complete_target_copy", "targets": []},
            {"state": "complete_quality_grid", "cells": []},
            {"benchmark_stage_sha256": "3" * 64},
            {"analysis_record_sha256": "2" * 64},
            {},
            {"source_manifest_sha256": "1" * 64},
        )


def test_replay_phase_a_failure_prevents_target_stage_and_phase_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "artifact"
    outdir.mkdir()
    (outdir / "source_snapshot.tar").write_bytes(b"placeholder")
    preflight = {
        "preflight_binding_sha256": "0" * 64,
        "source_manifest_sha256": "1" * 64,
        "operational_upstream_roots": {"comp008": str(tmp_path / "comp008")},
    }
    candidate = {"state": "complete_candidate_set"}
    authority = {
        "comp008": {
            "targets": [{"relative_path": "development/target.png"}]
        }
    }
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(run, "verify_preflight", lambda _outdir: preflight)
    monkeypatch.setattr(run, "verify_import_streams", lambda _outdir: {})
    monkeypatch.setattr(run, "verify_candidate_stage", lambda _outdir: candidate)
    monkeypatch.setattr(run, "_authority_copy", lambda *_args: authority)
    monkeypatch.setattr(run, "_read_canonical_object", lambda _path: {})
    monkeypatch.setattr(run, "safe_extract_source_archive", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        run,
        "_verify_runtime_source_tree",
        lambda *_args, **_kwargs: {"source_manifest_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        run,
        "_derive_replay_captured_source_authority",
        lambda *_args, **_kwargs: {"test": "source-authority"},
    )

    def fake_worker(*_args: object, **kwargs: object) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        calls.append(kwargs)
        return {}, {}, {}

    monkeypatch.setattr(run, "_run_json_worker", fake_worker)
    monkeypatch.setattr(
        run,
        "_verify_replay_phase_a",
        lambda *_args: (_ for _ in ()).throw(run.LifecycleError("phase A mismatch")),
    )
    monkeypatch.setattr(
        run,
        "_stage_guard",
        lambda *_args: pytest.fail("target-bearing predecessor verification ran after failed phase A"),
    )
    with pytest.raises(run.LifecycleError, match="phase A mismatch"):
        run.replay_stage(outdir)
    assert len(calls) == 1
    readonly = {Path(path) for path in calls[0]["read_only"]}
    assert outdir / "targets" not in readonly
    assert calls[0]["denied_read_probes"] == [
        tmp_path / "comp008/development/target.png",
        outdir / f"targets/{actual.FROZEN_IMAGES[0]}/target.png",
        outdir / f"targets/{actual.FROZEN_IMAGES[0]}/target.rgb",
    ]


def test_shallow_stage_chain_never_calls_target_or_quality_verifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records: dict[str, dict[str, object]] = {
        "preflight": {
            "preflight_binding_sha256": "0" * 64,
            "journal": {"length": 1, "head_sha256": "1" * 64},
        }
    }
    monkeypatch.setattr(run, "verify_preflight", lambda _outdir: records["preflight"])

    def shallow(
        _outdir: Path,
        *,
        stage: str,
        schema: str,
        prior_record: dict[str, object],
    ) -> dict[str, object]:
        del schema, prior_record
        value = {
            run._STAGE_SEAL_FIELDS[stage]: hashlib.sha256(stage.encode()).hexdigest(),
            "journal": {"length": 1, "head_sha256": "1" * 64},
        }
        records[stage] = value
        return value

    monkeypatch.setattr(run, "_verify_stage_base", shallow)
    monkeypatch.setattr(
        run, "verify_target_copy", lambda *_args, **_kwargs: pytest.fail("target verifier called")
    )
    monkeypatch.setattr(
        run, "verify_quality_stage", lambda *_args, **_kwargs: pytest.fail("quality verifier called")
    )
    observed = run._verified_stage_records(tmp_path, "replay")
    assert list(observed) == list(run.STAGES[:-1])
