from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import signal
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from benchmarks import ssp2v_actual_coder as actual
from benchmarks import ssp2v_actual_run as run
from benchmarks import ssp2v_landlock as sandbox


def _outer_source_denied_probe() -> Path:
    """Return one captured-source file intentionally omitted from inner grants."""

    path = run.ROOT / "tasks/COMP-011-complete-stream-rgb-vq.md"
    if not path.is_file():
        raise AssertionError("captured outer-source denied probe is absent")
    return path


def _reseal_attestation(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    result.pop("attestation_sha256", None)
    result["attestation_sha256"] = hashlib.sha256(
        run._canonical_json(result)  # noqa: SLF001 - exercise the persisted seal grammar.
    ).hexdigest()
    return result


def _attestation_skeleton() -> dict[str, Any]:
    command = [sys.executable, "-c", "pass"]
    environment: dict[str, str] = {}
    launcher = Path(sandbox.__file__).resolve(strict=True)
    return _reseal_attestation(
        {
            "schema": sandbox.ATTESTATION_SCHEMA,
            "pid": 424_242,
            "parent_pid": 1,
            "restriction_mode": "fresh_root",
            "parent_sandbox_attestation_sha256": None,
            "parent_death_signal": int(signal.SIGKILL),
            "parent_death_signal_verified": True,
            "parent_pid_race_checked": True,
            "landlock_abi": 6,
            "filesystem_restricted": True,
            "tcp_bind_connect_restricted": True,
            "non_unix_socket_creation_restricted": True,
            "external_unix_ipc_restricted": True,
            "cross_process_restricted": True,
            "landlock_scope": ["abstract_unix_socket", "signal"],
            "seccomp_denied_syscalls": sorted(sandbox._SECCOMP_DENIED_SYSCALLS),  # noqa: SLF001
            "read_only": [],
            "read_write": [],
            "denied_read_probes": [],
            "socket_probe": {
                "basis": "direct_bind_connect",
                "inherited_AF_INET_create": None,
                "inherited_AF_INET6_create": None,
                "AF_INET": errno.EACCES,
                "AF_INET6": errno.EACCES,
                "AF_UNIX_allowed": 1,
                "TCP_bind": errno.EACCES,
                "TCP_connect": errno.EACCES,
            },
            "command": command,
            "command_sha256": hashlib.sha256(run._canonical_json(command)).hexdigest(),  # noqa: SLF001
            "cwd": os.fspath(run.ROOT.resolve(strict=True)),
            "environment_keys": [],
            "environment_sha256_before_attestation_marker": hashlib.sha256(
                run._canonical_json(environment)  # noqa: SLF001
            ).hexdigest(),
            "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
        }
    )


_FOCUSED_SOURCE_MANIFEST_SHA256 = "f" * 64


def _focused_policy_environment(scratch: Path) -> dict[str, str]:
    source_root = scratch.parent / "source"
    dependencies_root = scratch.parent / "dependencies"
    return run._focused_proof_environment(  # noqa: SLF001 - exact policy oracle under test.
        source_root,
        dependencies_root,
        scratch,
        source_manifest_sha256=_FOCUSED_SOURCE_MANIFEST_SHA256,
    )


def _focused_environment(scratch: Path, nonce: str) -> dict[str, str]:
    environment = _focused_policy_environment(scratch)
    environment["STRUCTSPLAT_PARENT_LAUNCH_NONCE"] = nonce
    return environment


def _seal_worker_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    return actual.seal_record(value, "worker_policy_sha256")


def _focused_base(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in run._FOCUSED_BASE_FIELDS}  # noqa: SLF001


def _rebuild_focused_launch(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    result["sandbox_attestation"] = _reseal_attestation(result["sandbox_attestation"])
    policy = _seal_worker_policy(result["launch"]["worker_policy"])
    result["launch"] = run._launch_receipt(  # noqa: SLF001 - persisted launch evidence under test.
        nonce=str(result["launch"]["nonce"]),
        command=result["command"],
        worker=_focused_base(result),
        worker_seal_field="canonical_worker_sha256",
        attestation=result["sandbox_attestation"],
        worker_policy=policy,
    )
    return result


def _focused_record(workspace: Path, probes: list[Path]) -> dict[str, Any]:
    source_root = workspace / "source"
    dependencies_root = workspace / "dependencies"
    scratch = workspace / "scratch"
    base_temp = scratch / "pytest"
    command = run._focused_proof_command(base_temp)  # noqa: SLF001
    nonce = "a" * 64
    stdout = b"synthetic focused proof passed\n"
    stderr = b""
    dependency_record = actual.seal_record(
        {
            "schema": run.FOCUSED_DEPENDENCY_SCHEMA,
            "files": [
                {
                    "path": "torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
                    "bytes": 244_408_911,
                    "sha256": "e" * 64,
                }
            ],
            "network_download_permitted": False,
        },
        "focused_dependency_manifest_sha256",
    )
    base = {
        "command": command,
        "returncode": 0,
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_hex": stdout.hex(),
        "stderr_bytes": len(stderr),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_hex": stderr.hex(),
        "wall_ns": 1,
        "captured_source_manifest_sha256": _FOCUSED_SOURCE_MANIFEST_SHA256,
        "captured_source_file_count": 71,
        "captured_source_reverified_after_proof": True,
        "captured_runtime_dependencies": dependency_record,
        "captured_dependencies_reverified_after_proof": True,
    }
    attestation = _attestation_skeleton()
    policy_environment = _focused_policy_environment(scratch)
    environment = {**policy_environment, "STRUCTSPLAT_PARENT_LAUNCH_NONCE": nonce}
    child_pid = int(attestation["pid"])
    live_launcher = Path(sandbox.__file__).resolve(strict=True)
    launcher = source_root / "benchmarks/ssp2v_landlock.py"
    read_only_request = sorted(
        {
            (
                os.fspath(path)
                if path in {source_root, dependencies_root}
                else run._worker_policy_path(path)  # noqa: SLF001
            )
            for path in run._focused_proof_read_only(  # noqa: SLF001
                source_root, dependencies_root
            )
        }
    )
    read_write_request = sorted(
        {
            os.fspath(scratch)
            if path == scratch
            else run._worker_policy_path(path)  # noqa: SLF001
            for path in run._focused_proof_read_write(scratch)  # noqa: SLF001
        }
    )
    denied_paths = [os.fspath(path) for path in sorted(probes)]
    policy = _seal_worker_policy(
        {
            "schema": run.WORKER_POLICY_SCHEMA,
            "command": command,
            "cwd": os.fspath(source_root / "tests"),
            "environment": policy_environment,
            "read_only_request": read_only_request,
            "read_write_request": read_write_request,
            "denied_read_probe_paths": denied_paths,
            "launcher_path": os.fspath(launcher),
            "launcher_sha256": hashlib.sha256(live_launcher.read_bytes()).hexdigest(),
            "timeout_seconds": 1800.0,
        }
    )
    attestation.update(
        {
            "command": command,
            "command_sha256": hashlib.sha256(run._canonical_json(command)).hexdigest(),  # noqa: SLF001
            "cwd": os.fspath(source_root / "tests"),
            "environment_keys": sorted(environment),
            "environment_sha256_before_attestation_marker": hashlib.sha256(
                run._canonical_json(environment)  # noqa: SLF001
            ).hexdigest(),
            "launcher_sha256": hashlib.sha256(live_launcher.read_bytes()).hexdigest(),
            "read_only": sorted(
                {
                    run._attested_allowlist_path(path, child_pid)  # noqa: SLF001
                    for path in run._focused_proof_read_only(  # noqa: SLF001
                        source_root, dependencies_root
                    )
                }
            ),
            "read_write": sorted(
                {
                    run._attested_allowlist_path(path, child_pid)  # noqa: SLF001
                    for path in run._focused_proof_read_write(scratch)  # noqa: SLF001
                }
            ),
            "denied_read_probes": [
                {"path": os.fspath(path), "errno": errno.EACCES}
                for path in sorted(probes)
            ],
        }
    )
    attestation = _reseal_attestation(attestation)
    launch = run._launch_receipt(  # noqa: SLF001
        nonce=nonce,
        command=command,
        worker=base,
        worker_seal_field="canonical_worker_sha256",
        attestation=attestation,
        worker_policy=policy,
    )
    return {**base, "sandbox_attestation": attestation, "launch": launch}


@pytest.fixture
def focused_policy(tmp_path: Path) -> tuple[dict[str, Any], list[Path]]:
    workspace = Path(tempfile.gettempdir()) / (
        f"comp011-focused-suite-policy-{os.getpid()}-{tmp_path.name}"
    )
    probes = [
        tmp_path / "synthetic-upstream/development/stream.sspl1",
        tmp_path / "synthetic-upstream/development/target.png",
    ]
    return _focused_record(workspace, probes), probes


def test_exact_focused_proof_policy_accepts_its_frozen_request(
    focused_policy: tuple[dict[str, Any], list[Path]],
) -> None:
    record, probes = focused_policy
    run._verify_focused_proof_policy(record, expected_probe_paths=probes)  # noqa: SLF001


def test_exact_focused_proof_policy_replays_under_captured_source_marker(
    focused_policy: tuple[dict[str, Any], list[Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, probes = focused_policy
    outdir, _manifest, _runtime_source = _runtime_source_fixture(tmp_path)
    captured_root = outdir / run._RUNTIME_SOURCE_RELATIVE  # noqa: SLF001
    monkeypatch.setattr(run, "ROOT", captured_root)
    monkeypatch.setenv(  # noqa: SLF001
        run._CAPTURED_REPLAY_SOURCE_ENV, os.fspath(captured_root)
    )

    run._verify_focused_proof_policy(  # noqa: SLF001
        record,
        expected_probe_paths=probes,
        require_live_inventory=False,
    )


def test_exact_focused_proof_policy_replay_does_not_require_dynamic_tree_enumeration(
    focused_policy: tuple[dict[str, Any], list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, probes = focused_policy
    monkeypatch.setattr(
        run,
        "_sandbox_system_read_only",
        lambda: pytest.fail("focused replay re-enumerated live read-only paths"),
    )
    monkeypatch.setattr(
        run,
        "_sandbox_gpu_read_write",
        lambda: pytest.fail("focused replay re-enumerated live GPU paths"),
    )

    run._verify_focused_proof_policy(  # noqa: SLF001
        record,
        expected_probe_paths=probes,
        require_live_inventory=False,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "broad-system-read",
        "safe-shaped-system-read",
        "noncanonical-alias",
        "missing-source-read",
        "missing-fixed-read",
    ],
)
def test_exact_focused_proof_policy_rejects_resealed_path_capability_drift(
    focused_policy: tuple[dict[str, Any], list[Path]],
    mutation: str,
) -> None:
    record, probes = focused_policy
    hostile = copy.deepcopy(record)
    policy = hostile["launch"]["worker_policy"]
    attestation = hostile["sandbox_attestation"]
    source_root = Path(policy["cwd"]).parent
    if mutation in {
        "broad-system-read",
        "safe-shaped-system-read",
        "noncanonical-alias",
    }:
        path = {
            "broad-system-read": "/sys",
            "safe-shaped-system-read": (
                "/sys/devices/system/cpu/cpu999999/online"
            ),
            "noncanonical-alias": (
                "/sys/devices/system/cpu/cpu1/../cpu1/online"
            ),
        }[mutation]
        policy["read_only_request"] = sorted(
            {*policy["read_only_request"], path}
        )
        attestation["read_only"] = sorted({*attestation["read_only"], path})
    elif mutation == "missing-source-read":
        path = os.fspath(source_root)
        policy["read_only_request"].remove(path)
        attestation["read_only"].remove(path)
    else:
        path = "/usr"
        policy["read_only_request"].remove(path)
        attestation["read_only"].remove(path)
    hostile = _rebuild_focused_launch(hostile)

    with pytest.raises(
        run.LifecycleError,
        match="path policy inventory|path capabilities",
    ):
        run._verify_focused_proof_policy(hostile, expected_probe_paths=probes)  # noqa: SLF001
    if mutation in {
        "noncanonical-alias",
        "missing-source-read",
        "missing-fixed-read",
    }:
        with pytest.raises(
            run.LifecycleError,
            match="path policy inventory|path capabilities",
        ):
            run._verify_focused_proof_policy(  # noqa: SLF001
                hostile,
                expected_probe_paths=probes,
                require_live_inventory=False,
            )


@pytest.mark.parametrize("value", [None, "0" * 64])
def test_exact_focused_proof_policy_rejects_source_marker_drift(
    focused_policy: tuple[dict[str, Any], list[Path]],
    value: str | None,
) -> None:
    record, probes = focused_policy
    hostile = copy.deepcopy(record)
    policy_environment = hostile["launch"]["worker_policy"]["environment"]
    if value is None:
        policy_environment.pop("STRUCTSPLAT_SOURCE_MANIFEST_SHA256")
    else:
        policy_environment["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = value
    attested_environment = dict(policy_environment)
    attested_environment["STRUCTSPLAT_PARENT_LAUNCH_NONCE"] = str(
        hostile["launch"]["nonce"]
    )
    hostile["sandbox_attestation"]["environment_keys"] = sorted(
        attested_environment
    )
    hostile["sandbox_attestation"][
        "environment_sha256_before_attestation_marker"
    ] = hashlib.sha256(run._canonical_json(attested_environment)).hexdigest()  # noqa: SLF001
    hostile = _rebuild_focused_launch(hostile)

    with pytest.raises(run.LifecycleError, match="exact frozen policy"):
        run._verify_focused_proof_policy(hostile, expected_probe_paths=probes)  # noqa: SLF001


def test_sandbox_ancestry_requires_exact_root_or_immediate_parent() -> None:
    root = _attestation_skeleton()
    run._verify_sandbox_ancestry(  # noqa: SLF001
        root,
        expected_parent_sandbox_attestation_sha256=None,
    )
    nested = copy.deepcopy(root)
    nested["restriction_mode"] = "inherited_nested"
    nested["parent_sandbox_attestation_sha256"] = root["attestation_sha256"]
    nested["socket_probe"] = {
        "basis": "inherited_non_unix_creation_denial",
        "inherited_AF_INET_create": errno.EACCES,
        "inherited_AF_INET6_create": errno.EACCES,
        "TCP_bind": None,
        "TCP_connect": None,
        "AF_INET": errno.EACCES,
        "AF_INET6": errno.EACCES,
        "AF_UNIX_allowed": 1,
    }
    nested = _reseal_attestation(nested)
    run._verify_sandbox_ancestry(  # noqa: SLF001
        nested,
        expected_parent_sandbox_attestation_sha256=str(
            root["attestation_sha256"]
        ),
    )
    with pytest.raises(run.LifecycleError, match="immediate parent"):
        run._verify_sandbox_ancestry(  # noqa: SLF001
            nested,
            expected_parent_sandbox_attestation_sha256="0" * 64,
        )
    with pytest.raises(run.LifecycleError, match="fresh root"):
        run._verify_sandbox_ancestry(  # noqa: SLF001
            nested,
            expected_parent_sandbox_attestation_sha256=None,
        )


def _focused_metric_record(record: Mapping[str, Any]) -> dict[str, Any]:
    item = record["captured_runtime_dependencies"]["files"][0]
    return {
        "alexnet_relative_path": f"runtime/metrics/{item['path']}",
        "alexnet_bytes": item["bytes"],
        "alexnet_sha256": item["sha256"],
        "network_download_permitted": False,
    }


def test_focused_dependency_identity_cross_binds_to_persisted_metric_copy(
    focused_policy: tuple[dict[str, Any], list[Path]],
) -> None:
    record, _probes = focused_policy
    run._verify_focused_metric_dependency_binding(  # noqa: SLF001
        record, _focused_metric_record(record)
    )


def test_focused_source_file_count_binds_to_persisted_manifest(
    focused_policy: tuple[dict[str, Any], list[Path]],
) -> None:
    record, probes = focused_policy
    hostile = copy.deepcopy(record)
    hostile["captured_source_file_count"] += 1
    hostile = _rebuild_focused_launch(hostile)
    with pytest.raises(run.LifecycleError, match="private-workspace drift"):
        run._verify_focused_proof_policy(  # noqa: SLF001
            hostile,
            expected_probe_paths=probes,
            expected_source_file_count=record["captured_source_file_count"],
        )


@pytest.mark.parametrize(
    "mutation",
    ["hash", "bytes", "path", "schema", "network", "reverified"],
)
def test_focused_dependency_binding_rejects_coherently_resealed_drift(
    focused_policy: tuple[dict[str, Any], list[Path]],
    mutation: str,
) -> None:
    record, probes = focused_policy
    hostile = copy.deepcopy(record)
    metrics = _focused_metric_record(record)
    dependency = hostile["captured_runtime_dependencies"]
    if mutation == "hash":
        dependency["files"][0]["sha256"] = "0" * 64
    elif mutation == "bytes":
        dependency["files"][0]["bytes"] += 1
    elif mutation == "path":
        dependency["files"][0]["path"] = "torch/hub/checkpoints/foreign.pth"
    elif mutation == "schema":
        dependency["schema"] = "structsplat.comp011.focused-dependencies.hostile"
    elif mutation == "network":
        dependency["network_download_permitted"] = True
    else:
        hostile["captured_dependencies_reverified_after_proof"] = False
    if mutation != "reverified":
        hostile["captured_runtime_dependencies"] = actual.seal_record(
            dependency, "focused_dependency_manifest_sha256"
        )
    hostile = _rebuild_focused_launch(hostile)
    if mutation in {"hash", "bytes"}:
        run._verify_focused_proof_policy(  # noqa: SLF001
            hostile, expected_probe_paths=probes
        )
        with pytest.raises(run.LifecycleError, match="dependency identities differ"):
            run._verify_focused_metric_dependency_binding(hostile, metrics)  # noqa: SLF001
        return
    with pytest.raises((run.LifecycleError, sandbox.SandboxError)):
        run._verify_focused_proof_policy(  # noqa: SLF001
            hostile, expected_probe_paths=probes
        )


def _mutate_drop_test(record: dict[str, Any], _probes: list[Path], _tmp_path: Path) -> None:
    record["command"].remove(  # noqa: SLF001
        Path(run._FOCUSED_TESTS[-1]).name
    )
    record["sandbox_attestation"]["command"] = list(record["command"])
    record["sandbox_attestation"]["command_sha256"] = hashlib.sha256(
        run._canonical_json(record["command"])  # noqa: SLF001
    ).hexdigest()
    record["launch"]["worker_policy"]["command"] = list(record["command"])


def _mutate_cwd(record: dict[str, Any], _probes: list[Path], tmp_path: Path) -> None:
    record["sandbox_attestation"]["cwd"] = os.fspath(tmp_path)
    record["launch"]["worker_policy"]["cwd"] = os.fspath(tmp_path)


def _mutate_environment(record: dict[str, Any], _probes: list[Path], _tmp_path: Path) -> None:
    command = record["command"]
    workspace = Path(command[command.index("--basetemp") + 1]).parent
    environment = _focused_environment(workspace, str(record["launch"]["nonce"]))
    environment["LD_AUDIT"] = "/tmp/foreign-audit.so"
    record["sandbox_attestation"]["environment_keys"] = sorted(environment)
    record["sandbox_attestation"]["environment_sha256_before_attestation_marker"] = (
        hashlib.sha256(run._canonical_json(environment)).hexdigest()  # noqa: SLF001
    )
    policy_environment = dict(environment)
    policy_environment.pop("STRUCTSPLAT_PARENT_LAUNCH_NONCE")
    record["launch"]["worker_policy"]["environment"] = policy_environment


def _mutate_read_only(record: dict[str, Any], _probes: list[Path], tmp_path: Path) -> None:
    path = os.fspath(tmp_path / "foreign-read")
    record["sandbox_attestation"]["read_only"] = sorted(
        {*record["sandbox_attestation"]["read_only"], path}
    )
    record["launch"]["worker_policy"]["read_only_request"] = sorted(
        {*record["launch"]["worker_policy"]["read_only_request"], path}
    )


def _mutate_read_write(record: dict[str, Any], _probes: list[Path], tmp_path: Path) -> None:
    path = os.fspath(tmp_path / "foreign-write")
    record["sandbox_attestation"]["read_write"] = sorted(
        {*record["sandbox_attestation"]["read_write"], path}
    )
    record["launch"]["worker_policy"]["read_write_request"] = sorted(
        {*record["launch"]["worker_policy"]["read_write_request"], path}
    )


def _mutate_launcher(record: dict[str, Any], _probes: list[Path], _tmp_path: Path) -> None:
    record["sandbox_attestation"]["launcher_sha256"] = "0" * 64
    record["launch"]["worker_policy"]["launcher_sha256"] = "0" * 64


def _mutate_denied_missing(record: dict[str, Any], _probes: list[Path], _tmp_path: Path) -> None:
    record["sandbox_attestation"]["denied_read_probes"] = []
    record["launch"]["worker_policy"]["denied_read_probe_paths"] = []


def _mutate_denied_replaced(record: dict[str, Any], _probes: list[Path], tmp_path: Path) -> None:
    record["sandbox_attestation"]["denied_read_probes"][0]["path"] = os.fspath(
        tmp_path / "different-probe"
    )
    record["sandbox_attestation"]["denied_read_probes"].sort(key=lambda item: item["path"])
    record["launch"]["worker_policy"]["denied_read_probe_paths"] = [
        item["path"] for item in record["sandbox_attestation"]["denied_read_probes"]
    ]


def _mutate_denied_extra(record: dict[str, Any], _probes: list[Path], tmp_path: Path) -> None:
    record["sandbox_attestation"]["denied_read_probes"].append(
        {"path": os.fspath(tmp_path / "extra-probe"), "errno": errno.EACCES}
    )
    record["sandbox_attestation"]["denied_read_probes"].sort(key=lambda item: item["path"])
    record["launch"]["worker_policy"]["denied_read_probe_paths"] = [
        item["path"] for item in record["sandbox_attestation"]["denied_read_probes"]
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        _mutate_drop_test,
        _mutate_cwd,
        _mutate_environment,
        _mutate_read_only,
        _mutate_read_write,
        _mutate_launcher,
        _mutate_denied_missing,
        _mutate_denied_replaced,
        _mutate_denied_extra,
    ],
    ids=[
        "drop-focused-test",
        "cwd",
        "environment",
        "read-only",
        "read-write",
        "launcher",
        "denied-missing",
        "denied-replaced",
        "denied-extra",
    ],
)
def test_exact_focused_proof_policy_rejects_coherently_resealed_mutations(
    focused_policy: tuple[dict[str, Any], list[Path]],
    tmp_path: Path,
    mutation: Callable[[dict[str, Any], list[Path], Path], None],
) -> None:
    record, probes = focused_policy
    hostile = copy.deepcopy(record)
    mutation(hostile, probes, tmp_path)
    hostile = _rebuild_focused_launch(hostile)
    with pytest.raises((run.LifecycleError, sandbox.SandboxError)):
        run._verify_focused_proof_policy(hostile, expected_probe_paths=probes)  # noqa: SLF001


def _move_focused_workspace(record: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    hostile = copy.deepcopy(dict(record))
    source_root = workspace / "source"
    scratch = workspace / "scratch"
    command = hostile["command"]
    index = command.index("--basetemp")
    command[index + 1] = os.fspath(scratch / "pytest")
    attestation = hostile["sandbox_attestation"]
    attestation["command"] = list(command)
    attestation["command_sha256"] = hashlib.sha256(
        run._canonical_json(command)  # noqa: SLF001
    ).hexdigest()
    policy = hostile["launch"]["worker_policy"]
    old_scratch = Path(policy["environment"]["TMPDIR"])
    old_source = Path(policy["cwd"]).parent
    old_dependencies = old_scratch.parent / "dependencies"
    policy["command"] = list(command)
    policy["cwd"] = os.fspath(source_root / "tests")
    policy["environment"] = _focused_policy_environment(scratch)
    policy["read_only_request"] = sorted(
        value.replace(os.fspath(old_source), os.fspath(source_root)).replace(
            os.fspath(old_dependencies), os.fspath(workspace / "dependencies")
        )
        for value in policy["read_only_request"]
    )
    policy["read_write_request"] = sorted(
        os.fspath(scratch) if value == os.fspath(old_scratch) else value
        for value in policy["read_write_request"]
    )
    policy["launcher_path"] = os.fspath(
        source_root / "benchmarks/ssp2v_landlock.py"
    )
    attestation["cwd"] = os.fspath(source_root / "tests")
    attestation["read_only"] = sorted(
        value.replace(os.fspath(old_source), os.fspath(source_root)).replace(
            os.fspath(old_dependencies), os.fspath(workspace / "dependencies")
        )
        for value in attestation["read_only"]
    )
    environment = _focused_environment(scratch, str(hostile["launch"]["nonce"]))
    attestation["environment_keys"] = sorted(environment)
    attestation["environment_sha256_before_attestation_marker"] = hashlib.sha256(
        run._canonical_json(environment)  # noqa: SLF001
    ).hexdigest()
    child_pid = int(attestation["pid"])
    attestation["read_write"] = sorted(
        {
            run._attested_allowlist_path(path, child_pid)  # noqa: SLF001
            for path in run._focused_proof_read_write(scratch)  # noqa: SLF001
        }
    )
    return _rebuild_focused_launch(hostile)


@pytest.mark.parametrize("location", ["repository", "outdir", "upstream", "denied"])
def test_exact_focused_proof_policy_rejects_workspace_under_protected_roots(
    focused_policy: tuple[dict[str, Any], list[Path]],
    tmp_path: Path,
    location: str,
) -> None:
    record, probes = focused_policy
    roots = {
        "repository": run.ROOT,
        "outdir": tmp_path / "artifact",
        "upstream": tmp_path / "synthetic-upstream",
        "denied": probes[0],
    }
    workspace = roots[location] / f"comp011-focused-suite-{location}"
    hostile = _move_focused_workspace(record, workspace)
    with pytest.raises((run.LifecycleError, sandbox.SandboxError)):
        run._verify_focused_proof_policy(  # noqa: SLF001
            hostile,
            expected_probe_paths=probes,
            forbidden_roots=[
                run.ROOT,
                roots["outdir"],
                roots["upstream"],
                *probes,
            ],
        )


def test_focused_sources_are_unique_and_in_the_explicit_closure() -> None:
    this_test = Path(__file__).resolve(strict=True).relative_to(run.ROOT).as_posix()
    assert len(run._FOCUSED_TESTS) == len(set(run._FOCUSED_TESTS))  # noqa: SLF001
    assert set(run._FOCUSED_TESTS).issubset(run._EXPLICIT_SOURCES)  # noqa: SLF001
    assert this_test in run._FOCUSED_TESTS  # noqa: SLF001
    assert this_test in run._EXPLICIT_SOURCES  # noqa: SLF001
    assert all(
        (run.ROOT / relative).is_file() and not (run.ROOT / relative).is_symlink()
        for relative in run._FOCUSED_TESTS  # noqa: SLF001
    )


def test_focused_command_runs_recursive_sandbox_nodes_and_skips_only_host_probes(
    tmp_path: Path,
) -> None:
    command = run._focused_proof_command(tmp_path / "pytest")  # noqa: SLF001
    host_probes = (
        "test_ssp2v_actual_run.py"
        "::test_live_environment_binding_is_immediately_repeatable",
        "test_ssp2v_actual_run.py"
        "::test_bound_hardware_environment_replays_without_gpu_device_authority",
    )
    assert run._FOCUSED_HOST_PROBE_DESELECTS == host_probes  # noqa: SLF001
    deselect_args = [item for item in command if item.startswith("--deselect=")]
    deselects = {
        item.removeprefix("--deselect=")
        for item in deselect_args
    }
    assert deselects == {
        "test_ssp2v_actual_run.py"
        "::test_upstream_binding_opens_only_sealed_metadata_and_captured_source",
        *host_probes,
    }
    assert len(deselect_args) == len(deselects) == 3


def _runtime_source_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    outdir = tmp_path / "artifact"
    outdir.mkdir()
    manifest = run.source_manifest()
    archive = outdir / "source_snapshot.tar"
    run.write_source_archive(archive, manifest)
    source_root = outdir / run._RUNTIME_SOURCE_RELATIVE  # noqa: SLF001
    assert run.safe_extract_source_archive(  # noqa: SLF001
        archive, source_root, expected_manifest=manifest
    ) == manifest
    record = run._runtime_source_record(  # noqa: SLF001
        source_root,
        outdir=outdir,
        manifest=manifest,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    return outdir, manifest, record


def test_captured_replay_helpers_use_receipt_bound_frozen_path_inventories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir, _manifest, _record = _runtime_source_fixture(tmp_path)
    source_root = outdir / run._RUNTIME_SOURCE_RELATIVE  # noqa: SLF001
    inventory = run._replay_path_inventory_environment()  # noqa: SLF001
    system_values = json.loads(
        inventory[run._REPLAY_SYSTEM_READ_ONLY_ENV]  # noqa: SLF001
    )
    synthetic_dynamic_leaf = "/sys/devices/system/cpu/cpu999999/online"
    system_values = sorted({*system_values, synthetic_dynamic_leaf})
    gpu_values = ["/proc/self/task"]

    monkeypatch.setattr(run, "ROOT", source_root)
    monkeypatch.setenv(
        run._CAPTURED_REPLAY_SOURCE_ENV, os.fspath(source_root)  # noqa: SLF001
    )
    monkeypatch.setenv(run._REPLAY_PREFLIGHT_BINDING_ENV, "a" * 64)  # noqa: SLF001
    monkeypatch.setenv(sandbox.SANDBOX_ATTESTATION_ENV, "b" * 64)
    monkeypatch.setenv(
        run._REPLAY_SYSTEM_READ_ONLY_ENV,  # noqa: SLF001
        run._canonical_json(system_values).decode().removesuffix("\n"),  # noqa: SLF001
    )
    monkeypatch.setenv(
        run._REPLAY_GPU_READ_WRITE_ENV,  # noqa: SLF001
        run._canonical_json(gpu_values).decode().removesuffix("\n"),  # noqa: SLF001
    )

    assert synthetic_dynamic_leaf in {
        path.as_posix() for path in run._sandbox_system_read_only()  # noqa: SLF001
    }
    assert run._sandbox_gpu_read_write() == [Path("/proc/self/task")]  # noqa: SLF001


def test_captured_replay_path_inventory_rejects_broad_system_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir, _manifest, _record = _runtime_source_fixture(tmp_path)
    source_root = outdir / run._RUNTIME_SOURCE_RELATIVE  # noqa: SLF001
    inventory = run._replay_path_inventory_environment()  # noqa: SLF001
    system_values = sorted(
        {
            *json.loads(
                inventory[run._REPLAY_SYSTEM_READ_ONLY_ENV]  # noqa: SLF001
            ),
            "/sys",
        }
    )

    monkeypatch.setattr(run, "ROOT", source_root)
    monkeypatch.setenv(
        run._CAPTURED_REPLAY_SOURCE_ENV, os.fspath(source_root)  # noqa: SLF001
    )
    monkeypatch.setenv(run._REPLAY_PREFLIGHT_BINDING_ENV, "a" * 64)  # noqa: SLF001
    monkeypatch.setenv(sandbox.SANDBOX_ATTESTATION_ENV, "b" * 64)
    monkeypatch.setenv(
        run._REPLAY_SYSTEM_READ_ONLY_ENV,  # noqa: SLF001
        run._canonical_json(system_values).decode().removesuffix("\n"),  # noqa: SLF001
    )
    monkeypatch.setenv(
        run._REPLAY_GPU_READ_WRITE_ENV,  # noqa: SLF001
        inventory[run._REPLAY_GPU_READ_WRITE_ENV],  # noqa: SLF001
    )

    with pytest.raises(run.LifecycleError, match="exceeds frozen grammar"):
        run._sandbox_system_read_only()  # noqa: SLF001


def test_nested_landlock_reuses_full_frozen_replay_inventory(
    tmp_path: Path,
) -> None:
    parent_attestation = os.environ.get(sandbox.SANDBOX_ATTESTATION_ENV)
    expected_outer_mode = (
        sandbox.RESTRICTION_MODE_FRESH
        if parent_attestation is None
        else sandbox.RESTRICTION_MODE_INHERITED
    )
    outdir, _manifest, _record = _runtime_source_fixture(tmp_path)
    source_root = outdir / run._RUNTIME_SOURCE_RELATIVE  # noqa: SLF001
    system_read_only = run._sandbox_system_read_only()  # noqa: SLF001
    gpu_read_write = run._sandbox_gpu_read_write()  # noqa: SLF001
    inventory = run._replay_path_inventory_environment(  # noqa: SLF001
        system_read_only=system_read_only,
        gpu_read_write=gpu_read_write,
    )
    code = r'''
import os
import pathlib
import sys

root = pathlib.Path(os.environ["STRUCTSPLAT_SMOKE_SOURCE"])
os.environ["COMP011_CAPTURED_REPLAY_SOURCE_ROOT"] = str(root)
sys.path[:0] = [str(root), str(root / "src")]
from benchmarks import ssp2v_actual_run as run

system = run._sandbox_system_read_only()
gpu = run._sandbox_gpu_read_write()
nested_environment = dict(os.environ)
nested_environment.pop(run.sandbox.SANDBOX_ATTESTATION_ENV, None)
completed, attestation = run.sandbox.run_sandboxed(
    [sys.executable, "-I", "-B", "-c", "print('nested-frozen-inventory-ok')"],
    read_only=[*system, root],
    read_write=gpu,
    cwd=root / "benchmarks",
    env=nested_environment,
    timeout=60,
    launcher_path=root / "benchmarks/ssp2v_landlock.py",
)
if completed.returncode != 0:
    sys.stderr.buffer.write(completed.stderr)
    raise SystemExit(completed.returncode)
if attestation["restriction_mode"] != "inherited_nested":
    raise RuntimeError("nested sandbox did not inherit its outer Landlock policy")
sys.stdout.buffer.write(completed.stdout)
'''
    environment = os.environ.copy()
    environment.pop(sandbox.SANDBOX_ATTESTATION_ENV, None)
    environment.update(inventory)
    environment.update(
        {
            run._REPLAY_PREFLIGHT_BINDING_ENV: "a" * 64,  # noqa: SLF001
            "STRUCTSPLAT_SMOKE_SOURCE": os.fspath(source_root),
        }
    )
    completed, attestation = sandbox.run_sandboxed(
        [sys.executable, "-I", "-B", "-c", code],
        read_only=[*system_read_only, source_root],
        read_write=gpu_read_write,
        cwd=source_root / "benchmarks",
        env=environment,
        timeout=120,
        launcher_path=source_root / "benchmarks/ssp2v_landlock.py",
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert completed.stdout == b"nested-frozen-inventory-ok\n"
    assert attestation["restriction_mode"] == expected_outer_mode
    assert (
        attestation["parent_sandbox_attestation_sha256"]
        == parent_attestation
    )


def _preflight_worker_bundle(
    *,
    name: str,
    outdir: Path,
    scratch: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    source_root = outdir / run._RUNTIME_SOURCE_RELATIVE  # noqa: SLF001
    lloyd_path = outdir / "runtime/native/libssp2v_lloyd_v1.so"
    renderer_record_path = outdir / "runtime/renderer/record.json"
    lloyd_path.parent.mkdir(parents=True, exist_ok=True)
    if not lloyd_path.is_file():
        lloyd_path.write_bytes(b"synthetic-lloyd")
        os.chmod(lloyd_path, 0o444)
    renderer_record_path.parent.mkdir(parents=True, exist_ok=True)
    if not renderer_record_path.is_file():
        renderer_record_path.write_bytes(b"{}\n")
        os.chmod(renderer_record_path, 0o444)

    environment = run._worker_environment(  # noqa: SLF001
        outdir, source_root=source_root
    )
    read_only = [
        *run._sandbox_system_read_only(),  # noqa: SLF001
        *run._worker_source_read_only(source_root),  # noqa: SLF001
    ]
    read_write: list[Path] = []
    ephemeral: list[Path] = []
    if name == "lloyd_proof":
        command = [
            sys.executable,
            "-m",
            "benchmarks.ssp2v_actual_run",
            "_lloyd-proof",
            "--native",
            os.fspath(lloyd_path),
        ]
        read_only.append(lloyd_path)
        timeout = 360.0
        seal_field = "lloyd_proof_sha256"
        worker = {seal_field: "a" * 64}
    elif name == "renderer_build":
        command = [
            sys.executable,
            "-m",
            "benchmarks.ssp2v_actual_run",
            "_build-renderer",
        ]
        environment.update(
            {
                "TMPDIR": os.fspath(scratch),
                "TORCH_EXTENSIONS_DIR": os.fspath(scratch / "extensions"),
            }
        )
        read_write = [scratch, *run._sandbox_gpu_read_write()]  # noqa: SLF001
        ephemeral = [scratch]
        timeout = 1800.0
        seal_field = "renderer_build_worker_sha256"
        worker = {
            "extension_path": os.fspath(scratch / "extensions/module/renderer.so"),
            seal_field: "b" * 64,
        }
    else:
        command = [
            sys.executable,
            "-m",
            "benchmarks.ssp2v_actual_run",
            "_renderer-proof",
            "--artifact-root",
            os.fspath(outdir),
            "--renderer-record",
            os.fspath(renderer_record_path),
        ]
        environment["TMPDIR"] = os.fspath(scratch)
        read_only.append(outdir / "runtime")
        read_write = [scratch, *run._sandbox_gpu_read_write()]  # noqa: SLF001
        ephemeral = [scratch]
        timeout = 1800.0
        seal_field = "renderer_proof_worker_sha256"
        worker = {seal_field: "c" * 64}

    policy = run._expected_preflight_worker_policy(  # noqa: SLF001
        command=command,
        cwd=source_root / "benchmarks",
        environment=environment,
        read_only=read_only,
        read_write=read_write,
        ephemeral_paths=ephemeral,
        launcher_path=source_root / "benchmarks/ssp2v_landlock.py",
        timeout=timeout,
    )
    nonce = "d" * 64
    attestation = _attestation_skeleton()
    child_environment = {**environment, "STRUCTSPLAT_PARENT_LAUNCH_NONCE": nonce}
    child_pid = int(attestation["pid"])
    attestation.update(
        {
            "command": command,
            "command_sha256": hashlib.sha256(run._canonical_json(command)).hexdigest(),  # noqa: SLF001
            "cwd": os.fspath(source_root / "benchmarks"),
            "environment_keys": sorted(child_environment),
            "environment_sha256_before_attestation_marker": hashlib.sha256(
                run._canonical_json(child_environment)  # noqa: SLF001
            ).hexdigest(),
            "read_only": run._materialize_producer_request_paths(  # noqa: SLF001
                policy["read_only_request"], child_pid=child_pid
            ),
            "read_write": run._materialize_producer_request_paths(  # noqa: SLF001
                policy["read_write_request"], child_pid=child_pid
            ),
            "denied_read_probes": [],
            "launcher_sha256": policy["launcher_sha256"],
        }
    )
    attestation = _reseal_attestation(attestation)
    launch = run._launch_receipt(  # noqa: SLF001
        nonce=nonce,
        command=command,
        worker=worker,
        worker_seal_field=seal_field,
        attestation=attestation,
        worker_policy=policy,
    )
    return {"attestation": attestation, "launch": launch}, worker, lloyd_path, renderer_record_path


def _coherently_reseal_preflight_bundle(
    bundle: Mapping[str, Any], worker: Mapping[str, Any], seal_field: str
) -> dict[str, Any]:
    hostile = copy.deepcopy(dict(bundle))
    policy = actual.seal_record(
        hostile["launch"]["worker_policy"], "worker_policy_sha256"
    )
    nonce = str(hostile["launch"]["nonce"])
    attestation = hostile["attestation"]
    environment = {
        **policy["environment"],
        "STRUCTSPLAT_PARENT_LAUNCH_NONCE": nonce,
    }
    child_pid = int(attestation["pid"])
    attestation.update(
        {
            "command": policy["command"],
            "command_sha256": hashlib.sha256(
                run._canonical_json(policy["command"])  # noqa: SLF001
            ).hexdigest(),
            "cwd": policy["cwd"],
            "environment_keys": sorted(environment),
            "environment_sha256_before_attestation_marker": hashlib.sha256(
                run._canonical_json(environment)  # noqa: SLF001
            ).hexdigest(),
            "read_only": run._materialize_producer_request_paths(  # noqa: SLF001
                policy["read_only_request"], child_pid=child_pid
            ),
            "read_write": run._materialize_producer_request_paths(  # noqa: SLF001
                policy["read_write_request"], child_pid=child_pid
            ),
            "denied_read_probes": [
                {"path": value, "errno": errno.EACCES}
                for value in policy["denied_read_probe_paths"]
            ],
            "launcher_sha256": policy["launcher_sha256"],
        }
    )
    attestation = _reseal_attestation(attestation)
    launch = run._launch_receipt(  # noqa: SLF001
        nonce=nonce,
        command=policy["command"],
        worker=worker,
        worker_seal_field=seal_field,
        attestation=attestation,
        worker_policy=policy,
    )
    return {"attestation": attestation, "launch": launch}


def test_runtime_source_replica_imports_without_live_repository_authority(
    tmp_path: Path,
) -> None:
    outdir, manifest, runtime_source = _runtime_source_fixture(tmp_path)
    source_root = outdir / run._RUNTIME_SOURCE_RELATIVE  # noqa: SLF001
    denied_probe = _outer_source_denied_probe()
    code = """
from benchmarks import ssp2v_actual_run as run
record = run.actual.seal_record({
    "module_root": str(run.ROOT),
    "source_manifest_sha256": run.source_manifest()["source_manifest_sha256"],
}, "probe_sha256")
run.sys.stdout.buffer.write(run._canonical_json(record))
"""
    worker, attestation, launch = run._run_json_worker(  # noqa: SLF001
        [sys.executable, "-B", "-c", code],
        outdir=outdir,
        timeout=60,
        denied_read_probes=[denied_probe],
        worker_seal_field="probe_sha256",
    )
    assert worker["module_root"] == os.fspath(source_root)
    assert worker["source_manifest_sha256"] == manifest["source_manifest_sha256"]
    assert attestation["denied_read_probes"] == [
        {"path": os.fspath(denied_probe), "errno": errno.EACCES}
    ]
    bundle = {"attestation": attestation, "launch": launch}
    run._verify_launch_bundle(  # noqa: SLF001
        bundle,
        worker,
        "probe_sha256",
        expected_parent_sandbox_attestation_sha256=os.environ.get(
            sandbox.SANDBOX_ATTESTATION_ENV
        ),
    )
    run._verify_runtime_source_worker_policy(  # noqa: SLF001
        bundle,
        outdir=outdir,
        runtime_source=runtime_source,
        manifest=manifest,
        archive_sha256=hashlib.sha256(
            (outdir / "source_snapshot.tar").read_bytes()
        ).hexdigest(),
    )


def test_live_checkout_marker_cannot_override_ordinary_runtime_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir, _manifest, _record = _runtime_source_fixture(tmp_path)
    monkeypatch.setenv(run._CAPTURED_REPLAY_SOURCE_ENV, os.fspath(run.ROOT))  # noqa: SLF001
    with pytest.raises(run.LifecycleError, match="source-root marker"):
        run._worker_source_root(outdir)  # noqa: SLF001


def test_captured_marker_requires_and_accepts_only_its_verified_module_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir, _manifest, _record = _runtime_source_fixture(tmp_path)
    source_root = outdir / run._RUNTIME_SOURCE_RELATIVE  # noqa: SLF001
    monkeypatch.setattr(run, "ROOT", source_root)
    monkeypatch.setenv(run._CAPTURED_REPLAY_SOURCE_ENV, os.fspath(source_root))  # noqa: SLF001
    assert run._worker_source_root(outdir) == source_root  # noqa: SLF001


def test_exact_preflight_worker_policies_reject_coherently_resealed_authority_drift(
    tmp_path: Path,
) -> None:
    outdir, manifest, runtime_source = _runtime_source_fixture(tmp_path)
    archive_sha256 = hashlib.sha256(
        (outdir / "source_snapshot.tar").read_bytes()
    ).hexdigest()
    suffixes = {
        "lloyd_proof": "lloyd_proof_sha256",
        "renderer_build": "renderer_build_worker_sha256",
        "renderer_proof": "renderer_proof_worker_sha256",
    }
    prefixes = {
        "lloyd_proof": "comp011-unused-",
        "renderer_build": "comp011-renderer-build-",
        "renderer_proof": "comp011-renderer-proof-",
    }
    for name, seal_field in suffixes.items():
        scratch = Path("/tmp") / f"{prefixes[name]}{hashlib.sha256(name.encode()).hexdigest()[:8]}"
        assert not os.path.lexists(scratch)
        bundle, worker, lloyd_path, renderer_record_path = _preflight_worker_bundle(
            name=name,
            outdir=outdir,
            scratch=scratch,
        )
        run._verify_launch_bundle(bundle, worker, seal_field)  # noqa: SLF001
        run._verify_exact_preflight_worker_policy(  # noqa: SLF001
            name,
            bundle,
            worker,
            outdir=outdir,
            runtime_source=runtime_source,
            manifest=manifest,
            archive_sha256=archive_sha256,
            lloyd_path=lloyd_path,
            renderer_record_path=renderer_record_path,
            forbidden_roots=[run.ROOT, outdir],
        )

        mutations = [
            "command",
            "cwd",
            "environment-extra",
            "environment-preload",
            "read-only",
            "read-only-missing-runtime",
            "read-write",
            "denied",
            "launcher",
            "timeout",
        ]
        if name != "lloyd_proof":
            mutations.append("scratch-grammar")
        for mutation in mutations:
            hostile = copy.deepcopy(bundle)
            policy = hostile["launch"]["worker_policy"]
            if mutation == "command":
                policy["command"] = [*policy["command"], "--hostile"]
            elif mutation == "cwd":
                policy["cwd"] = "/tmp"
            elif mutation == "environment-extra":
                policy["environment"]["LD_AUDIT"] = "/tmp/hostile-audit.so"
            elif mutation == "environment-preload":
                policy["environment"]["LD_PRELOAD"] = "/tmp/hostile-preload.so"
            elif mutation == "read-only":
                policy["read_only_request"] = sorted(
                    {
                        *policy["read_only_request"],
                        os.fspath(_outer_source_denied_probe()),
                    }
                )
            elif mutation == "read-only-missing-runtime":
                policy["read_only_request"].remove(
                    run._worker_policy_path(Path("/dev/null"))  # noqa: SLF001
                )
            elif mutation == "read-write":
                policy["read_write_request"] = sorted(
                    {*policy["read_write_request"], os.fspath(outdir / run._RUNTIME_SOURCE_RELATIVE)}  # noqa: SLF001
                )
            elif mutation == "denied":
                policy["denied_read_probe_paths"] = [
                    os.fspath(_outer_source_denied_probe())
                ]
            elif mutation == "launcher":
                policy["launcher_sha256"] = "0" * 64
            elif mutation == "timeout":
                policy["timeout_seconds"] = float(policy["timeout_seconds"]) + 1.0
            else:
                policy["environment"]["TMPDIR"] = "/tmp/foreign-worker-12345678"
            hostile = _coherently_reseal_preflight_bundle(
                hostile, worker, seal_field
            )
            run._verify_launch_bundle(hostile, worker, seal_field)  # noqa: SLF001
            with pytest.raises(run.LifecycleError):
                run._verify_exact_preflight_worker_policy(  # noqa: SLF001
                    name,
                    hostile,
                    worker,
                    outdir=outdir,
                    runtime_source=runtime_source,
                    manifest=manifest,
                    archive_sha256=archive_sha256,
                    lloyd_path=lloyd_path,
                    renderer_record_path=renderer_record_path,
                    forbidden_roots=[run.ROOT, outdir],
                )


def test_runtime_source_executes_exact_lloyd_proof_worker(tmp_path: Path) -> None:
    outdir, manifest, runtime_source = _runtime_source_fixture(tmp_path)
    native = outdir / "runtime/native/libssp2v_lloyd_v1.so"
    run.lloyd.build_native(native)
    os.chmod(native, 0o444)
    denied_probe = _outer_source_denied_probe()
    worker, attestation, launch = run._run_json_worker(  # noqa: SLF001
        [
            sys.executable,
            "-m",
            "benchmarks.ssp2v_actual_run",
            "_lloyd-proof",
            "--native",
            os.fspath(native),
        ],
        outdir=outdir,
        timeout=360,
        read_only=[native],
        denied_read_probes=[denied_probe],
        worker_seal_field="lloyd_proof_sha256",
    )
    run.validate_lloyd_proof(worker)
    bundle = {"attestation": attestation, "launch": launch}
    run._verify_launch_bundle(  # noqa: SLF001
        bundle,
        worker,
        "lloyd_proof_sha256",
        expected_parent_sandbox_attestation_sha256=os.environ.get(
            sandbox.SANDBOX_ATTESTATION_ENV
        ),
    )
    run._verify_runtime_source_worker_policy(  # noqa: SLF001
        bundle,
        outdir=outdir,
        runtime_source=runtime_source,
        manifest=manifest,
        archive_sha256=hashlib.sha256(
            (outdir / "source_snapshot.tar").read_bytes()
        ).hexdigest(),
    )


def test_sandboxed_ninja_executes_one_edge_with_exact_system_authority(
    tmp_path: Path,
) -> None:
    """Exercise Ninja's real posix_spawn stdin action under the worker policy."""

    ninja = Path("/usr/bin/ninja")
    if not ninja.is_file():
        pytest.skip("the COMP-011 renderer build requires /usr/bin/ninja")
    scratch = tmp_path / "ninja"
    scratch.mkdir()
    (scratch / "build.ninja").write_text(
        "rule touch\n"
        "  command = /usr/bin/touch $out\n"
        "\n"
        "build spawn.ok: touch\n"
        "default spawn.ok\n",
        encoding="utf-8",
    )
    profile_alias = Path("/usr/lib/nvidia-cuda-toolkit/bin/nvcc.profile")
    code = r"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ninja, scratch, profile_alias = sys.argv[1:]
completed = subprocess.run(
    [ninja, "-v", "-C", scratch],
    check=False,
    capture_output=True,
    timeout=30,
)
if completed.returncode != 0 or completed.stderr:
    sys.stderr.buffer.write(completed.stdout)
    sys.stderr.buffer.write(completed.stderr)
    raise SystemExit(completed.returncode or 1)
profile_sha256 = None
if profile_alias != "-":
    profile_sha256 = hashlib.sha256(Path(profile_alias).read_bytes()).hexdigest()
record = {
    "ninja_returncode": completed.returncode,
    "output_created": (Path(scratch) / "spawn.ok").is_file(),
    "profile_sha256": profile_sha256,
}
sys.stdout.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
"""
    worker, attestation, launch = run._run_json_worker(  # noqa: SLF001
        [
            sys.executable,
            "-B",
            "-c",
            code,
            os.fspath(ninja),
            os.fspath(scratch),
            os.fspath(profile_alias) if profile_alias.is_file() else "-",
        ],
        outdir=None,
        timeout=60,
        read_write=[scratch],
        worker_seal_field="canonical_worker_sha256",
        cwd=scratch,
        include_repository_source=False,
    )
    assert worker == {
        "ninja_returncode": 0,
        "output_created": True,
        "profile_sha256": (
            hashlib.sha256(profile_alias.read_bytes()).hexdigest()
            if profile_alias.is_file()
            else None
        ),
    }
    run._verify_launch_bundle(  # noqa: SLF001
        {"attestation": attestation, "launch": launch},
        worker,
        "canonical_worker_sha256",
        expected_parent_sandbox_attestation_sha256=os.environ.get(
            sandbox.SANDBOX_ATTESTATION_ENV
        ),
    )
    expected_runtime_reads = {
        run._worker_policy_path(path)  # noqa: SLF001
        for path in (Path("/dev/null"), Path("/etc/nvcc.profile"))
        if path.exists()
    }
    policy_reads = set(launch["worker_policy"]["read_only_request"])
    attested_reads = set(attestation["read_only"])
    assert expected_runtime_reads <= policy_reads
    assert expected_runtime_reads <= attested_reads


@pytest.mark.parametrize(
    "path",
    [Path("/dev/null"), Path("/etc/nvcc.profile")],
    ids=["ninja-stdin", "nvcc-profile"],
)
def test_system_policy_classifier_accepts_exact_build_runtime_reads(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"optional build runtime path is absent: {path}")
    canonical = Path(run._worker_policy_path(path))  # noqa: SLF001
    assert run._safe_system_read_only_policy_path(canonical)  # noqa: SLF001
    assert not run._safe_system_read_only_policy_path(  # noqa: SLF001
        Path("/etc/passwd") if path.parent == Path("/etc") else Path("/dev/zero")
    )


@pytest.mark.parametrize(
    "path",
    [Path("/proc"), Path("/sys"), Path("/sys/devices/system/cpu")],
)
def test_system_policy_classifier_rejects_broad_host_tree_grants(path: Path) -> None:
    assert not run._safe_system_read_only_policy_path(path)  # noqa: SLF001


@pytest.mark.parametrize("mutation", ["writable-file", "extra-directory", "record-hash"])
def test_runtime_source_replica_rejects_authority_drift(
    tmp_path: Path, mutation: str
) -> None:
    outdir, manifest, record = _runtime_source_fixture(tmp_path)
    source_root = outdir / run._RUNTIME_SOURCE_RELATIVE  # noqa: SLF001
    archive_sha256 = hashlib.sha256(
        (outdir / "source_snapshot.tar").read_bytes()
    ).hexdigest()
    if mutation == "writable-file":
        os.chmod(source_root / "benchmarks/ssp2v_actual_run.py", 0o644)
    elif mutation == "extra-directory":
        os.chmod(source_root, 0o755)
        (source_root / "foreign-empty-directory").mkdir(mode=0o555)
        os.chmod(source_root, 0o555)
    else:
        record = copy.deepcopy(record)
        record["source_manifest_sha256"] = "0" * 64
        record = actual.seal_record(record, "runtime_source_sha256")
    with pytest.raises(run.LifecycleError):
        run._verify_runtime_source_record(  # noqa: SLF001
            outdir,
            record,
            expected_manifest=manifest,
            expected_archive_sha256=archive_sha256,
        )


@pytest.mark.parametrize("hostility", ["duplicate", "missing-explicit"])
def test_focused_proof_fails_before_launch_on_source_inventory_drift(
    monkeypatch: pytest.MonkeyPatch,
    hostility: str,
) -> None:
    first = run._FOCUSED_TESTS[0]  # noqa: SLF001
    if hostility == "duplicate":
        monkeypatch.setattr(run, "_FOCUSED_TESTS", (*run._FOCUSED_TESTS, first))
    else:
        monkeypatch.setattr(
            run,
            "_EXPLICIT_SOURCES",
            tuple(path for path in run._EXPLICIT_SOURCES if path != first),
        )
    monkeypatch.setattr(
        run.sandbox,
        "run_sandboxed",
        lambda *_args, **_kwargs: pytest.fail("invalid source inventory reached a worker launch"),
    )
    with pytest.raises(run.LifecycleError, match="source closure"):
        run.focused_proof_suite(denied_read_probes=[])


def _import_parse_bundle(
    *,
    outdir: Path,
    authority: Mapping[str, Any],
    payload: bytes,
    source_path: Path,
    workspace: Path,
    denied_probe: Path,
    nonce: str,
) -> dict[str, Any]:
    parsed = {"canonical_cold_parse": True}
    config = run._import_worker_config(authority)  # noqa: SLF001
    worker = actual.seal_record(
        {
            "schema": "structsplat.comp011.import-stream-worker.v1",
            "protocol": run.PROTOCOL,
            "task_sha256": run.TASK_SHA256,
            "image_id": authority["image_id"],
            "bit_tuple": authority["bit_tuple"],
            "import_worker_config_sha256": config["import_worker_config_sha256"],
            "blob_bytes": len(payload),
            "blob_sha256": hashlib.sha256(payload).hexdigest(),
            "parsed": parsed,
            "target_or_pixel_payloads_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "import_worker_sha256",
    )
    command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_actual_run",
        "_import-stream",
        "--source",
        os.fspath(source_path),
        "--config",
        os.fspath(workspace / "config.json"),
    ]
    attestation = _attestation_skeleton()
    source_root = run._worker_source_root(outdir)  # noqa: SLF001
    policy_environment = run._worker_environment(  # noqa: SLF001
        outdir, source_root=source_root
    )
    environment = {
        **policy_environment,
        "STRUCTSPLAT_PARENT_LAUNCH_NONCE": nonce,
    }
    read_only_request = sorted(
        {
            run._worker_policy_path(path)  # noqa: SLF001
            for path in (
                *run._sandbox_system_read_only(),  # noqa: SLF001
                *run._worker_source_read_only(source_root),  # noqa: SLF001
                source_path,
            )
        }
    )
    read_write_request = sorted(
        {
            os.fspath(workspace),
            *(
                run._worker_policy_path(path)  # noqa: SLF001
                for path in run._sandbox_process_read_write()  # noqa: SLF001
            ),
        }
    )
    child_pid = int(attestation["pid"])
    attestation.update(
        {
            "command": command,
            "command_sha256": hashlib.sha256(run._canonical_json(command)).hexdigest(),  # noqa: SLF001
            "cwd": os.fspath((source_root / "benchmarks").resolve(strict=True)),
            "environment_keys": sorted(environment),
            "environment_sha256_before_attestation_marker": hashlib.sha256(
                run._canonical_json(environment)  # noqa: SLF001
            ).hexdigest(),
            "read_only": run._materialize_producer_request_paths(  # noqa: SLF001
                read_only_request,
                child_pid=child_pid,
            ),
            "read_write": run._materialize_producer_request_paths(  # noqa: SLF001
                read_write_request,
                child_pid=child_pid,
            ),
            "denied_read_probes": [
                {"path": os.fspath(denied_probe), "errno": errno.EACCES}
            ],
        }
    )
    attestation = _reseal_attestation(attestation)
    launcher = (source_root / "benchmarks/ssp2v_landlock.py").resolve(strict=True)
    policy = _seal_worker_policy(
        {
            "schema": run.WORKER_POLICY_SCHEMA,
            "command": command,
            "cwd": attestation["cwd"],
            "environment": policy_environment,
            "read_only_request": read_only_request,
            "read_write_request": read_write_request,
            "denied_read_probe_paths": [os.fspath(denied_probe)],
            "launcher_path": os.fspath(launcher),
            "launcher_sha256": attestation["launcher_sha256"],
            "timeout_seconds": 600.0,
        }
    )
    launch = run._launch_receipt(  # noqa: SLF001
        nonce=nonce,
        command=command,
        worker=worker,
        worker_seal_field="import_worker_sha256",
        attestation=attestation,
        worker_policy=policy,
    )
    return {
        "worker": worker,
        "sandbox_attestation": attestation,
        "launch": launch,
        "denied_read_probe_paths": [os.fspath(denied_probe)],
    }


def _rebuild_import_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(bundle))
    policy = result["launch"]["worker_policy"]
    policy = _seal_worker_policy(policy)
    result["worker"] = actual.seal_record(
        result["worker"],
        "import_worker_sha256",
    )
    attestation = result["sandbox_attestation"]
    nonce = str(result["launch"]["nonce"])
    environment = {
        **policy["environment"],
        "STRUCTSPLAT_PARENT_LAUNCH_NONCE": nonce,
    }
    child_pid = int(attestation["pid"])
    attestation.update(
        {
            "command": policy["command"],
            "command_sha256": hashlib.sha256(
                run._canonical_json(policy["command"])  # noqa: SLF001
            ).hexdigest(),
            "cwd": policy["cwd"],
            "environment_keys": sorted(environment),
            "environment_sha256_before_attestation_marker": hashlib.sha256(
                run._canonical_json(environment)  # noqa: SLF001
            ).hexdigest(),
            "read_only": run._materialize_producer_request_paths(  # noqa: SLF001
                policy["read_only_request"],
                child_pid=child_pid,
            ),
            "read_write": run._materialize_producer_request_paths(  # noqa: SLF001
                policy["read_write_request"],
                child_pid=child_pid,
            ),
            "denied_read_probes": [
                {"path": path, "errno": errno.EACCES}
                for path in policy["denied_read_probe_paths"]
            ],
            "launcher_sha256": policy["launcher_sha256"],
        }
    )
    attestation = _reseal_attestation(attestation)
    result["sandbox_attestation"] = attestation
    result["denied_read_probe_paths"] = policy["denied_read_probe_paths"]
    result["launch"] = run._launch_receipt(  # noqa: SLF001
        nonce=nonce,
        command=policy["command"],
        worker=result["worker"],
        worker_seal_field="import_worker_sha256",
        attestation=attestation,
        worker_policy=policy,
    )
    return result


@pytest.mark.parametrize(
    "hostility",
    [
        "command",
        "source",
        "config-path",
        "config-digest",
        "environment",
        "cwd",
        "read-only",
        "read-write",
        "launcher",
        "denied",
        "timeout",
    ],
)
def test_import_stream_verifier_rejects_coherently_sealed_exact_policy_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hostility: str,
) -> None:
    """Generic launch self-consistency must not substitute for an exact policy."""

    outdir = tmp_path / "artifact"
    upstream_root = tmp_path / "synthetic-upstream"
    denied_probe = upstream_root / "development/target-0.png"
    extra_target = upstream_root / "development/target-1.png"
    preflight_record = {
        "operational_upstream_roots": {
            "comp008": os.fspath(upstream_root),
            "comp009": os.fspath(tmp_path / "synthetic-comp009"),
            "comp010": os.fspath(tmp_path / "synthetic-comp010"),
        }
    }
    identities = run._expected_identities()  # noqa: SLF001
    streams: list[dict[str, Any]] = []
    authorities: list[dict[str, Any]] = []
    for index, (image_id, bits) in enumerate(identities):
        payload = f"synthetic-sspl1-{index}".encode()
        relative = Path("streams") / image_id / run._tuple_label(bits) / "sspl1.bin"  # noqa: SLF001
        path = outdir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o444)
        digest = hashlib.sha256(payload).hexdigest()
        symbol = hashlib.sha256(payload + b"-symbols").hexdigest()
        boundary = hashlib.sha256(payload + b"-boundary").hexdigest()
        cell_seal = hashlib.sha256(payload + b"-cell").hexdigest()
        cell = {
            "image_id": image_id,
            "bit_tuple": list(bits),
            "cell_record_sha256": cell_seal,
            "sspl1": {
                "path": f"development/{image_id}-{run._tuple_label(bits)}.sspl1",  # noqa: SLF001
                "bytes": len(payload),
                "sha256": digest,
            },
            "absolute_symbols": {"symbol_sha256": symbol},
            "decoded_boundary_state_sha256": boundary,
        }
        authorities.append(cell)
        streams.append(
            {
                "image_id": image_id,
                "bit_tuple": list(bits),
                "relative_path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": digest,
                "symbol_sha256": symbol,
                "boundary_state_sha256": boundary,
                "comp008_cell_record_sha256": cell_seal,
                "canonical_cold_parse": True,
                "parse_worker": _import_parse_bundle(
                    outdir=outdir,
                    authority=cell,
                    payload=payload,
                    source_path=path,
                    workspace=Path(tempfile.gettempdir())
                    / f"comp011-import-worker-synthetic-{os.getpid()}-{index}",
                    denied_probe=denied_probe,
                    nonce=f"{index + 1:064x}",
                ),
            }
        )
    authority = {
        "authority_manifest_sha256": "8" * 64,
        "comp008": {
            "cells": authorities,
            "targets": [
                {"relative_path": "development/target-0.png"},
                {"relative_path": "development/target-1.png"},
            ],
        },
    }
    stage_record = {
        "authority_manifest_sha256": authority["authority_manifest_sha256"],
        "streams": streams,
        "stream_count": 16,
        "development_streams_opened": True,
        "target_or_pixel_payloads_opened": False,
        "candidate_or_model_products_created": False,
    }
    monkeypatch.setattr(
        run,
        "verify_preflight",
        lambda *_args, **_kwargs: preflight_record,
    )
    monkeypatch.setattr(run, "_verify_stage_base", lambda *_args, **_kwargs: stage_record)
    monkeypatch.setattr(run, "_authority_copy", lambda *_args, **_kwargs: authority)

    assert run.verify_import_streams(outdir) is stage_record

    bundle = copy.deepcopy(streams[0]["parse_worker"])
    policy = bundle["launch"]["worker_policy"]
    if hostility == "command":
        policy["command"][3] = "_produce-cell"
    elif hostility == "source":
        policy["command"][5] = os.fspath(extra_target)
        policy["read_only_request"] = sorted(
            {*policy["read_only_request"], os.fspath(extra_target)}
        )
    elif hostility == "config-path":
        policy["command"][7] = os.fspath(
            Path(policy["command"][7]).with_name("other-config.json")
        )
    elif hostility == "config-digest":
        bundle["worker"]["import_worker_config_sha256"] = "9" * 64
    elif hostility == "environment":
        policy["environment"]["LD_AUDIT"] = "/tmp/foreign-audit.so"
    elif hostility == "cwd":
        policy["cwd"] = os.fspath(tmp_path)
    elif hostility == "read-only":
        policy["read_only_request"] = sorted(
            {*policy["read_only_request"], os.fspath(extra_target)}
        )
    elif hostility == "read-write":
        policy["read_write_request"] = sorted(
            {*policy["read_write_request"], os.fspath(tmp_path / "foreign-write")}
        )
    elif hostility == "launcher":
        policy["launcher_sha256"] = "0" * 64
    elif hostility == "denied":
        policy["denied_read_probe_paths"] = [os.fspath(extra_target)]
    else:
        policy["timeout_seconds"] = 601.0
    streams[0]["parse_worker"] = _rebuild_import_bundle(bundle)
    with pytest.raises(run.LifecycleError):
        run.verify_import_streams(outdir)


def test_import_stream_historical_policy_survives_captured_source_relocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay relocation must not rewrite the source root of an old launch receipt."""

    ordinary_root = tmp_path / "ordinary"
    captured_root = tmp_path / "captured"
    ordinary_root.mkdir()
    captured_root.mkdir()
    outdir, _manifest, _runtime_record = _runtime_source_fixture(ordinary_root)
    captured_outdir, _captured_manifest, _captured_record = _runtime_source_fixture(
        captured_root
    )
    original_source_root = outdir / run._RUNTIME_SOURCE_RELATIVE  # noqa: SLF001
    captured_source_root = (
        captured_outdir / run._RUNTIME_SOURCE_RELATIVE  # noqa: SLF001
    )

    upstream_root = tmp_path / "synthetic-upstream"
    artifact_source = outdir / "streams/source.sspl1"
    artifact_source.parent.mkdir(parents=True)
    artifact_source.write_bytes(b"synthetic-sspl1")
    artifact_source.chmod(0o444)
    denied_probe = upstream_root / "development/target-0.png"
    workspace = (
        Path(tempfile.gettempdir())
        / f"comp011-import-worker-relocation-{os.getpid()}"
    )
    authority = {
        "image_id": run._expected_identities()[0][0],  # noqa: SLF001
        "bit_tuple": list(run._expected_identities()[0][1]),  # noqa: SLF001
        "sspl1": {
            "path": "development/source.sspl1",
            "bytes": artifact_source.stat().st_size,
            "sha256": hashlib.sha256(artifact_source.read_bytes()).hexdigest(),
        },
        "absolute_symbols": {"symbol_sha256": "1" * 64},
        "decoded_boundary_state_sha256": "2" * 64,
        "cell_record_sha256": "3" * 64,
    }
    preflight_record = {
        "operational_upstream_roots": {
            "comp008": os.fspath(upstream_root),
            "comp009": os.fspath(tmp_path / "synthetic-comp009"),
            "comp010": os.fspath(tmp_path / "synthetic-comp010"),
        }
    }
    normalized_authority = {
        "comp008": {
            "targets": [{"relative_path": "development/target-0.png"}],
        }
    }
    bundle = _import_parse_bundle(
        outdir=outdir,
        authority=authority,
        payload=artifact_source.read_bytes(),
        source_path=artifact_source,
        workspace=workspace,
        denied_probe=denied_probe,
        nonce="4" * 64,
    )
    assert bundle["launch"]["worker_policy"]["cwd"] == os.fspath(
        original_source_root / "benchmarks"
    )

    monkeypatch.setattr(run, "ROOT", captured_source_root)
    monkeypatch.setenv(
        run._CAPTURED_REPLAY_SOURCE_ENV,  # noqa: SLF001
        os.fspath(captured_source_root),
    )
    run._verify_import_stream_launch_policy(  # noqa: SLF001
        outdir=outdir,
        preflight_record=preflight_record,
        normalized_authority=normalized_authority,
        expected_cell=authority,
        artifact_source=artifact_source,
        worker=bundle["worker"],
        attestation=bundle["sandbox_attestation"],
        launch=bundle["launch"],
        denied_read_probe_paths=bundle["denied_read_probe_paths"],
    )
def test_focused_proof_policy_grants_pytest_writable_devnull(tmp_path: Path) -> None:
    """Pytest opens os.devnull for both capture input and its logging file handler."""

    assert Path(os.devnull) in run._focused_proof_read_write(tmp_path)  # noqa: SLF001
    assert Path(os.devnull) not in run._focused_proof_read_only(  # noqa: SLF001
        tmp_path / "source"
    )


@pytest.mark.parametrize("broad_root", [Path("/proc"), Path("/sys/devices/system/cpu")])
def test_focused_proof_rejects_broad_host_probe_read_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broad_root: Path
) -> None:
    narrow = run._sandbox_system_read_only()  # noqa: SLF001
    monkeypatch.setattr(
        run,
        "_sandbox_system_read_only",
        lambda: [*narrow, broad_root],
    )
    with pytest.raises(run.LifecycleError, match="excessive host-probe"):
        run._focused_proof_read_only(  # noqa: SLF001
            tmp_path / "source", tmp_path / "dependencies"
        )
