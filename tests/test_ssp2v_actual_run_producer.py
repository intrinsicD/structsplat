from __future__ import annotations

import copy
import errno
import hashlib
import ast
import inspect
import json
import os
import signal
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from benchmarks import ssp2v_actual_coder as actual
from benchmarks import ssp2v_actual_run as run
from benchmarks import ssp2v_landlock as sandbox


def test_execute_cell_is_reachable_only_inside_the_confined_producer_worker() -> None:
    tree = ast.parse(inspect.getsource(run))
    owners: list[str] = []

    class ExecuteCellVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute_cell"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "execute"
            ):
                owners.append(self.functions[-1] if self.functions else "<module>")
            self.generic_visit(node)

    ExecuteCellVisitor().visit(tree)
    assert owners == ["produce_candidate_cell_worker"]


def _reseal_attestation(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    result.pop("attestation_sha256", None)
    result["attestation_sha256"] = hashlib.sha256(
        run._canonical_json(result)  # noqa: SLF001 - audit the production seal grammar.
    ).hexdigest()
    return result


@pytest.fixture(scope="module")
def attestation_template() -> dict[str, Any]:
    # These tests mutate and reseal parent-side policy evidence; launching a
    # second irreversible sandbox inside the focused proof would test the
    # inherited outer seccomp layer instead.  Live launcher behavior is covered
    # by test_ssp2v_landlock and by the focused proof's own attestation.
    environment: dict[str, str] = {}
    command = [sys.executable, "-I", "-B", "-c", "pass"]
    launcher = Path(sandbox.__file__).resolve(strict=True)
    attestation = _reseal_attestation(
        {
            "schema": sandbox.ATTESTATION_SCHEMA,
            "pid": 424_243,
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
            "seccomp_denied_syscalls": sorted(
                sandbox._SECCOMP_DENIED_SYSCALLS  # noqa: SLF001
            ),
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
            "command_sha256": hashlib.sha256(
                run._canonical_json(command)  # noqa: SLF001
            ).hexdigest(),
            "cwd": os.fspath((run.ROOT / "benchmarks").resolve(strict=True)),
            "environment_keys": [],
            "environment_sha256_before_attestation_marker": hashlib.sha256(
                run._canonical_json(environment)  # noqa: SLF001
            ).hexdigest(),
            "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
        }
    )
    sandbox.verify_attestation(attestation)
    return attestation


def test_captured_comp009_system_reads_are_filtered_from_exact_parent_policy() -> None:
    system = {
        run._worker_policy_path(path)  # noqa: SLF001 - exact policy grammar under test.
        for path in run._sandbox_system_read_only()  # noqa: SLF001
    }
    foreign = {
        "/tmp/foreign-read",
        run.ROOT.resolve(strict=True).as_posix(),
        (run.ROOT / "benchmarks").resolve(strict=True).as_posix(),
    }
    parent = sorted(system | foreign)

    carried = run._captured_comp009_system_read_only(parent)  # noqa: SLF001

    assert carried == sorted(system)
    assert not foreign.intersection(carried)
    for malformed in (
        ["relative/path"],
        ["/usr", "/usr"],
        ["/usr/lib", "/usr"],
        [1],
    ):
        with pytest.raises(run.LifecycleError):
            run._captured_comp009_system_read_only(malformed)  # type: ignore[arg-type]  # noqa: SLF001


def test_nested_comp009_carries_parent_system_leaves_without_rediscovery(
    tmp_path: Path,
) -> None:
    source_root = run.ROOT.resolve(strict=True)
    dynamic_candidates = [
        *(Path(f"/sys/devices/system/cpu/cpu{index}/online") for index in range(1, 257)),
        Path("/sys/module/nvidia"),
        Path("/sys/module/nvidia_drm"),
        Path("/sys/module/nvidia_modeset"),
        Path("/sys/module/nvidia_uvm"),
    ]
    parent_authorized_leaf = next(
        (path for path in dynamic_candidates if path.exists()),
        None,
    )
    assert parent_authorized_leaf is not None
    parent_system = sorted(
        {
            run._worker_policy_path(path)  # noqa: SLF001
            for path in run._sandbox_system_read_only()  # noqa: SLF001
        }
        | {
            run._worker_policy_path(parent_authorized_leaf)  # noqa: SLF001
        }
    )
    expected_sentinel = run._worker_policy_path(parent_authorized_leaf)  # noqa: SLF001
    inventory = tmp_path / "parent-system-read-only.json"
    inventory.write_text(json.dumps(parent_system), encoding="utf-8")
    script = r"""
import json
import pathlib
import sys

from benchmarks import ssp2v_actual_run as run
from benchmarks import ssp2v_landlock as sandbox

inventory = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
source_root = pathlib.Path(sys.argv[2]).resolve(strict=True)
sentinel = sys.argv[3]
carried = run._captured_comp009_system_read_only(inventory)
rediscovered = sorted({
    run._worker_policy_path(path)
    for path in run._sandbox_system_read_only()
})
missing_from_rediscovery = sorted(set(carried) - set(rediscovered))
if sentinel not in missing_from_rediscovery:
    raise RuntimeError("outer sandbox did not hide any sealed system leaf")
environment = run._worker_environment(source_root=source_root)
command = [
    sys.executable,
    "-I",
    "-B",
    "-c",
    "import pathlib,sys; pathlib.Path(sys.argv[1]).stat()",
    sentinel,
]
completed, attestation = sandbox.run_sandboxed(
    command,
    read_only=[*(pathlib.Path(value) for value in carried), source_root],
    read_write=run._sandbox_process_read_write(),
    cwd=source_root,
    env=environment,
    timeout=60,
    launcher_path=source_root / "benchmarks/ssp2v_landlock.py",
)
if completed.returncode != 0 or completed.stdout or completed.stderr:
    raise RuntimeError("nested system-leaf read failed")
print(json.dumps({
    "carried": carried,
    "missing_from_rediscovery": missing_from_rediscovery,
    "sentinel": sentinel,
    "nested_restriction_mode": attestation["restriction_mode"],
    "nested_parent": attestation["parent_sandbox_attestation_sha256"],
}, sort_keys=True))
"""
    environment = run._worker_environment(source_root=source_root)  # noqa: SLF001
    completed, attestation = sandbox.run_sandboxed(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            os.fspath(inventory),
            os.fspath(source_root),
            expected_sentinel,
        ],
        read_only=[
            *(Path(value) for value in parent_system),
            source_root,
            inventory,
        ],
        read_write=run._sandbox_process_read_write(),  # noqa: SLF001
        cwd=source_root,
        env=environment,
        timeout=120,
        launcher_path=source_root / "benchmarks/ssp2v_landlock.py",
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert completed.stderr == b""
    result = json.loads(completed.stdout)
    assert result["carried"] == parent_system
    assert result["missing_from_rediscovery"]
    assert result["sentinel"] == expected_sentinel
    assert expected_sentinel in result["missing_from_rediscovery"]
    assert result["nested_restriction_mode"] == sandbox.RESTRICTION_MODE_INHERITED
    assert result["nested_parent"] == attestation["attestation_sha256"]


def test_metric_private_tmpdir_supports_torch_lazy_import(
    tmp_path: Path,
) -> None:
    source_root = run.ROOT.resolve(strict=True)
    scratch = tmp_path / "metric-scratch"
    scratch.mkdir()
    environment = run._worker_environment(source_root=source_root)  # noqa: SLF001
    environment["TMPDIR"] = os.fspath(scratch)
    command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        (
            "import tempfile; "
            "from torch.distributed.nn.jit import instantiator; "
            "handle=tempfile.TemporaryDirectory(); handle.cleanup()"
        ),
    ]

    completed, attestation = sandbox.run_sandboxed(
        command,
        read_only=[*run._sandbox_system_read_only(), source_root],  # noqa: SLF001
        read_write=[scratch],
        cwd=source_root,
        env=environment,
        timeout=120,
        launcher_path=source_root / "benchmarks/ssp2v_landlock.py",
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert completed.stdout == b""
    assert completed.stderr == b""
    assert os.fspath(scratch.resolve(strict=True)) in attestation["read_write"]


def test_render_metric_orchestration_uses_and_cleans_private_metric_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "artifact"
    runtime = outdir / "runtime"
    runtime.mkdir(parents=True)
    renderer_record_path = runtime / "renderer-record.json"
    renderer_record_path.write_bytes(b"{}")
    blob_path = outdir / "candidate.ssp2v"
    blob_path.write_bytes(b"candidate")
    native_path = runtime / "native.so"
    native_path.write_bytes(b"native")
    target_path = outdir / "targets/image.rgb"
    target_path.parent.mkdir()
    target_path.write_bytes(b"\x00\x01\x02")

    temporary_paths: list[Path] = []
    calls: list[str] = []

    monkeypatch.setattr(run, "_worker_environment", lambda _outdir: {"BASE": "1"})
    monkeypatch.setattr(run, "_sandbox_gpu_read_write", lambda: [])

    def fake_worker(
        command: list[str],
        *,
        outdir: Path,
        timeout: int,
        read_only: list[Path],
        read_write: list[Path],
        worker_seal_field: str,
        environment: Mapping[str, str],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        assert outdir == tmp_path / "artifact"
        assert timeout == 600
        if worker_seal_field == "render_worker_sha256":
            calls.append("render")
            render_path = Path(command[command.index("--output") + 1])
            temporary = render_path.parent
            temporary_paths.append(temporary)
            assert temporary.name.startswith("comp011-render-metric-")
            assert temporary.is_dir()
            assert environment == {"BASE": "1", "TMPDIR": os.fspath(temporary)}
            assert read_write == [temporary]
            assert runtime in read_only
            assert blob_path in read_only
            assert not (temporary / "metric-scratch").exists()
            render_path.write_bytes(b"render")
            record = actual.seal_record(
                {
                    "schema": "structsplat.comp011.render-arm-worker.v1",
                    "render_npy": {"sha256": "1" * 64},
                    "render_tensor": {"sha256": "2" * 64},
                },
                worker_seal_field,
            )
        else:
            calls.append("metric")
            scratch = Path(environment["TMPDIR"])
            temporary_paths.append(scratch)
            assert scratch.name == "metric-scratch"
            assert scratch.is_dir()
            assert environment == {"BASE": "1", "TMPDIR": os.fspath(scratch)}
            assert read_write == [scratch]
            assert runtime in read_only
            assert target_path in read_only
            assert Path(command[command.index("--render") + 1]) in read_only
            record = actual.seal_record(
                {
                    "schema": "structsplat.comp011.metric-worker.v1",
                    "render_tensor": {"sha256": "2" * 64},
                    "fresh_single_thread_cpu_worker": True,
                },
                worker_seal_field,
            )
        return record, {"worker": calls[-1]}, {"worker": calls[-1]}

    monkeypatch.setattr(run, "_run_json_worker", fake_worker)
    task = run._run_render_metric_task(  # noqa: SLF001
        outdir=outdir,
        renderer_record_path=renderer_record_path,
        renderer_record={},
        blob_path=blob_path,
        blob_sha256="3" * 64,
        symbol_sha256="4" * 64,
        boundary_state_sha256="5" * 64,
        native_path=native_path,
        native_record={"bytes": 6, "sha256": "6" * 64},
        target_record={
            "rgb_copy": {
                "relative_path": target_path.relative_to(outdir).as_posix(),
                "sha256": "7" * 64,
            },
            "rgb_dimensions_hw": [1, 1],
        },
        expected_lpips_state_sha256="8" * 64,
    )

    assert calls == ["render", "metric"]
    assert task["render_sandbox"] == {"worker": "render"}
    assert task["metric_sandbox"] == {"worker": "metric"}
    assert all(not path.exists() for path in temporary_paths)


def _producer_command(
    *,
    outdir: Path,
    source: Path,
    captured: Path,
    native: Path,
    lloyd: Path,
    workspace: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_actual_run",
        "_produce-cell",
        "--artifact-root",
        os.fspath(outdir),
        "--source",
        os.fspath(source),
        "--captured-comp009-root",
        os.fspath(captured),
        "--native",
        os.fspath(native),
        "--lloyd",
        os.fspath(lloyd),
        "--config",
        os.fspath(workspace / "config.json"),
        "--output-root",
        os.fspath(workspace / "output"),
    ]


def _producer_attestation(
    template: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    nonce: str,
    parent_sandbox_attestation_sha256: str | None = None,
) -> dict[str, Any]:
    environment = dict(policy["environment"])
    environment["STRUCTSPLAT_PARENT_LAUNCH_NONCE"] = nonce
    result = copy.deepcopy(dict(template))
    result.update(
        {
            "command": list(policy["command"]),
            "command_sha256": hashlib.sha256(
                run._canonical_json(policy["command"])  # noqa: SLF001
            ).hexdigest(),
            "cwd": policy["cwd"],
            "environment_keys": sorted(environment),
            "environment_sha256_before_attestation_marker": hashlib.sha256(
                run._canonical_json(environment)  # noqa: SLF001
            ).hexdigest(),
            "launcher_sha256": policy["launcher_sha256"],
            "read_only": run._materialize_producer_request_paths(  # noqa: SLF001
                policy["read_only_request"], child_pid=template["pid"]
            ),
            "read_write": run._materialize_producer_request_paths(  # noqa: SLF001
                policy["read_write_request"], child_pid=template["pid"]
            ),
            "denied_read_probes": [
                {"path": path, "errno": errno.EACCES}
                for path in policy["denied_read_probe_paths"]
            ],
        }
    )
    if parent_sandbox_attestation_sha256 is not None:
        result.update(
            {
                "restriction_mode": "inherited_nested",
                "parent_sandbox_attestation_sha256": (
                    parent_sandbox_attestation_sha256
                ),
                "socket_probe": {
                    "basis": "inherited_non_unix_creation_denial",
                    "inherited_AF_INET_create": errno.EACCES,
                    "inherited_AF_INET6_create": errno.EACCES,
                    "TCP_bind": None,
                    "TCP_connect": None,
                    "AF_INET": errno.EACCES,
                    "AF_INET6": errno.EACCES,
                    "AF_UNIX_allowed": 1,
                },
            }
        )
    return _reseal_attestation(result)


def _launch(
    *,
    attestation: Mapping[str, Any],
    worker: Mapping[str, Any],
    seal_field: str,
    nonce: str,
    policy: Mapping[str, Any],
    rebind_environment: bool = True,
    timeout: float = 1.0,
) -> dict[str, Any]:
    if rebind_environment:
        bound_environment = dict(policy["environment"])
        bound_environment["STRUCTSPLAT_PARENT_LAUNCH_NONCE"] = nonce
        rebound = copy.deepcopy(dict(attestation))
        rebound["environment_keys"] = sorted(bound_environment)
        rebound["environment_sha256_before_attestation_marker"] = hashlib.sha256(
            run._canonical_json(bound_environment)  # noqa: SLF001
        ).hexdigest()
        rebound = _reseal_attestation(rebound)
        if isinstance(attestation, dict):
            attestation.clear()
            attestation.update(rebound)
        attestation = rebound
    worker_policy = actual.seal_record(
        {
            "schema": run.WORKER_POLICY_SCHEMA,
            "command": list(attestation["command"]),
            "cwd": attestation["cwd"],
            "environment": dict(policy["environment"]),
            "read_only_request": list(policy["read_only_request"]),
            "read_write_request": list(policy["read_write_request"]),
            "denied_read_probe_paths": [
                item["path"] for item in attestation["denied_read_probes"]
            ],
            "launcher_path": policy["launcher_path"],
            "launcher_sha256": policy["launcher_sha256"],
            "timeout_seconds": timeout,
        },
        "worker_policy_sha256",
    )
    return run._launch_receipt(  # noqa: SLF001 - persisted launch evidence is under test.
        nonce=nonce,
        command=attestation["command"],
        worker=worker,
        worker_seal_field=seal_field,
        attestation=attestation,
        worker_policy=worker_policy,
    )


def test_parent_routes_complete_candidate_production_through_exact_private_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attestation_template: Mapping[str, Any],
) -> None:
    outdir = tmp_path / "artifact"
    runtime = outdir / "runtime"
    captured = tmp_path / "captured-comp009"
    runtime.mkdir(parents=True)
    captured.mkdir()
    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    source = outdir / "streams" / image_id / run._tuple_label(bits) / "sspl1.bin"  # noqa: SLF001
    native = runtime / "native/arithmetic.so"
    lloyd = runtime / "native/lloyd.so"
    target = tmp_path / "existing-target.png"
    source.parent.mkdir(parents=True)
    native.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic stream")
    native.write_bytes(b"synthetic arithmetic")
    lloyd.write_bytes(b"synthetic lloyd")
    for path in (source, native, lloyd):
        path.chmod(0o444)
    target.write_bytes(b"must remain unread")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    monkeypatch.setattr(
        run.execute,
        "execute_cell",
        lambda *_args, **_kwargs: pytest.fail(
            "the unrestricted parent called execute_cell directly"
        ),
    )
    monkeypatch.setattr(run.execute, "verify_record", lambda _record: None)
    monkeypatch.setattr(
        run,
        "_verify_captured_comp009_reproduction_policy",
        lambda *_args, **_kwargs: None,
    )

    def fake_run_json_worker(
        command: list[str], **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        calls.append((list(command), dict(kwargs)))
        output = Path(command[command.index("--output-root") + 1])
        config = run._read_canonical_object(  # noqa: SLF001
            Path(command[command.index("--config") + 1])
        )
        output.joinpath("exact").mkdir(parents=True)
        payload = b"child-produced-only"
        child_artifact = output / "exact/sspl1.bin"
        child_artifact.write_bytes(payload)
        child_artifact.chmod(0o444)
        execution = {"status": "synthetic_complete"}
        policy = config["producer_policy"]
        worker = actual.seal_record(
            {
                "schema": "structsplat.comp011.producing-cell-worker.v1",
                "protocol": run.PROTOCOL,
                "task_sha256": run.TASK_SHA256,
                "image_id": image_id,
                "bit_tuple": list(bits),
                "producing_config_sha256": config["producing_config_sha256"],
                "producer_policy_sha256": policy["producer_policy_sha256"],
                "execution": execution,
                "artifacts": [
                    {
                        "path": "exact/sspl1.bin",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
                "diagnostics": {
                    "captured_comp009_reproduction": {},
                    "decision_use": False,
                },
                "artifact_root_identity": {
                    "native": native.relative_to(outdir).as_posix(),
                    "lloyd": lloyd.relative_to(outdir).as_posix(),
                },
                "target_or_pixel_payloads_opened": False,
                "confirmation_payloads_accessed": False,
            },
            "producing_worker_sha256",
        )
        nonce = "b" * 64
        attestation = _producer_attestation(
            attestation_template,
            policy=policy,
            nonce=nonce,
            parent_sandbox_attestation_sha256=os.environ.get(
                sandbox.SANDBOX_ATTESTATION_ENV
            ),
        )
        launch = _launch(
            attestation=attestation,
            worker=worker,
            seal_field="producing_worker_sha256",
            nonce=nonce,
            policy=policy,
        )
        return worker, attestation, launch

    monkeypatch.setattr(run, "_run_json_worker", fake_run_json_worker)
    source_record = {
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    native_record = {
        "path": "arithmetic.so",
        "bytes": native.stat().st_size,
        "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
    }
    lloyd_record = {
        "path": "lloyd.so",
        "bytes": lloyd.stat().st_size,
        "sha256": hashlib.sha256(lloyd.read_bytes()).hexdigest(),
    }
    result, diagnostics = run._run_producing_cell_worker(  # noqa: SLF001
        outdir=outdir,
        image_id=image_id,
        bits=bits,
        source_path=source,
        source_record=source_record,
        captured_comp009_root=captured,
        comp009_authority={},
        captured_source_manifest_sha256="c" * 64,
        native_path=native,
        native_record=native_record,
        lloyd_path=lloyd,
        lloyd_record=lloyd_record,
        denied_read_probes=[target],
    )

    assert result.baseline_blobs == {"sspl1": b"child-produced-only"}
    assert diagnostics["producing_worker"]["denied_read_probe_paths"] == [
        os.fspath(target)
    ]
    assert len(calls) == 1
    command, kwargs = calls[0]
    workspace = Path(command[command.index("--config") + 1]).parent
    assert command == _producer_command(
        outdir=outdir,
        source=source,
        captured=captured,
        native=native,
        lloyd=lloyd,
        workspace=workspace,
    )
    assert kwargs["read_only"] == [runtime, source, captured]
    assert kwargs["read_write"] == [workspace, *run._sandbox_process_read_write()]
    assert kwargs["denied_read_probes"] == [target]
    assert kwargs["worker_seal_field"] == "producing_worker_sha256"
    assert kwargs.get("include_repository_source", True) is True
    assert kwargs["environment"]["TMPDIR"] == os.fspath(workspace)
    assert outdir not in kwargs["read_write"]
    assert target.exists()


@pytest.fixture
def persisted_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attestation_template: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    outdir = tmp_path / "artifact"
    runtime = outdir / "runtime"
    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    stream = outdir / "streams" / image_id / run._tuple_label(bits) / "sspl1.bin"  # noqa: SLF001
    captured = tmp_path / "captured-comp009"
    workspace = tmp_path / "private-producer"
    target = tmp_path / "existing-target.png"
    native = runtime / "native/arithmetic.so"
    lloyd = runtime / "native/lloyd.so"
    for directory in (runtime, stream.parent, captured, workspace, native.parent):
        directory.mkdir(parents=True, exist_ok=True)
    for path, payload in (
        (stream, b"stream"),
        (target, b"target"),
        (native, b"native"),
        (lloyd, b"lloyd"),
    ):
        path.write_bytes(payload)
    for path in (stream, native, lloyd):
        path.chmod(0o444)

    base = outdir / run._candidate_cell_relative(image_id, bits)  # noqa: SLF001
    exact_paths = {f"exact/{name}.bin" for name in run.execute.EXACT_FORMAT_ORDER}
    variant_paths = {f"variants/{name}.ssp2v" for name in run.execute.VARIANT_LABELS}
    artifact_paths = sorted(exact_paths | variant_paths)
    artifacts: list[dict[str, Any]] = []
    for index, relative in enumerate(artifact_paths):
        payload = f"synthetic-{index}".encode()
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        run._exclusive_write(path, payload)  # noqa: SLF001
        artifacts.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    by_path = {item["path"]: item for item in artifacts}
    execution = {
        "status": "valid_complete_candidate_set",
        "exact_formats": {
            name: {
                "complete_bytes": by_path[f"exact/{name}.bin"]["bytes"],
                "blob_sha256": by_path[f"exact/{name}.bin"]["sha256"],
            }
            for name in run.execute.EXACT_FORMAT_ORDER
        },
        "variants": {
            name: {
                "stream": {
                    "complete_bytes": by_path[f"variants/{name}.ssp2v"]["bytes"],
                    "blob_sha256": by_path[f"variants/{name}.ssp2v"]["sha256"],
                }
            }
            for name in run.execute.VARIANT_LABELS
        },
    }
    worker_source = {"bytes": 7, "sha256": "2" * 64}
    native_authority = {
        "path": "arithmetic.so",
        "bytes": native.stat().st_size,
        "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
    }
    lloyd_authority = {
        "path": "lloyd.so",
        "bytes": lloyd.stat().st_size,
        "sha256": hashlib.sha256(lloyd.read_bytes()).hexdigest(),
    }
    comp009_authority = {
        "image_id": image_id,
        "bit_tuple": list(bits),
        "input": {
            "bytes": stream.stat().st_size,
            "sha256": hashlib.sha256(stream.read_bytes()).hexdigest(),
            "symbol_sha256": "d" * 64,
            "decoded_state_sha256": "e" * 64,
        },
    }
    source_bindings = {
        "worker_module": {
            "source_label": run.decode_worker.WORKER_MODULE_LABEL,
            "bytes": worker_source["bytes"],
            "sha256": worker_source["sha256"],
            "identity_verified_before_operation": True,
        },
        "native_library": {
            "artifact_relative_path": "runtime/native/arithmetic.so",
            "bytes": native_authority["bytes"],
            "sha256": native_authority["sha256"],
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
    policy = run._producer_launch_policy(  # noqa: SLF001
        outdir=outdir,
        source_path=stream,
        captured_comp009_root=captured,
        native_path=native,
        lloyd_path=lloyd,
        config_path=workspace / "config.json",
        output_root=workspace / "output",
        denied_read_probes=[target],
    )
    producer_config = actual.seal_record(
        {
            "schema": "structsplat.comp011.producing-cell-config.v1",
            "protocol": run.PROTOCOL,
            "task_sha256": run.TASK_SHA256,
            "image_id": image_id,
            "bit_tuple": list(bits),
            "source_bytes": stream.stat().st_size,
            "source_sha256": hashlib.sha256(stream.read_bytes()).hexdigest(),
            "native_record": native_authority,
            "lloyd_record": lloyd_authority,
            "comp009_authority": comp009_authority,
            "captured_source_manifest_sha256": "c" * 64,
            "producer_policy": policy,
        },
        "producing_config_sha256",
    )
    producing_worker = actual.seal_record(
        {
            "schema": "structsplat.comp011.producing-cell-worker.v1",
            "protocol": run.PROTOCOL,
            "task_sha256": run.TASK_SHA256,
            "image_id": image_id,
            "bit_tuple": list(bits),
            "producing_config_sha256": producer_config["producing_config_sha256"],
            "producer_policy_sha256": policy["producer_policy_sha256"],
            "execution": execution,
            "artifacts": artifacts,
            "diagnostics": {"decision_use": False},
            "artifact_root_identity": {
                "native": "runtime/native/arithmetic.so",
                "lloyd": "runtime/native/lloyd.so",
            },
            "target_or_pixel_payloads_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "producing_worker_sha256",
    )
    producing_attestation = _producer_attestation(
        attestation_template,
        policy=policy,
        nonce="4" * 64,
    )
    producing_launch = _launch(
        attestation=producing_attestation,
        worker=producing_worker,
        seal_field="producing_worker_sha256",
        nonce="4" * 64,
        policy=policy,
    )

    cold_attestation = copy.deepcopy(producing_attestation)
    cold_attestation.update(
        {
            "command": [sys.executable, "-m", "benchmarks.ssp2v_decode_worker"],
            "denied_read_probes": [],
        }
    )
    cold_attestation["command_sha256"] = hashlib.sha256(
        run._canonical_json(cold_attestation["command"])  # noqa: SLF001
    ).hexdigest()
    cold_attestation = _reseal_attestation(cold_attestation)
    cold_decodes: dict[str, Any] = {}
    for index, relative in enumerate(artifact_paths):
        cold_worker = {
            "blob": {
                "artifact_relative_path": (base / relative)
                .relative_to(outdir)
                .as_posix()
            },
            "source_bindings": source_bindings,
            "cold_seal_sha256": hashlib.sha256(f"cold-{index}".encode()).hexdigest(),
        }
        sample_attestation = copy.deepcopy(cold_attestation)
        cold_decodes[relative] = {
            "cold_seal": cold_worker,
            "sandbox_attestation": sample_attestation,
            "launch": _launch(
                attestation=sample_attestation,
                worker=cold_worker,
                seal_field="cold_seal_sha256",
                nonce=hashlib.sha256(f"nonce-{index}".encode()).hexdigest(),
                policy=policy,
            ),
        }

    nested_workspace = workspace / "comp011-comp009-cell-abcdefgh"
    nested_source = nested_workspace / "source.sspl1"
    nested_output_parent = nested_workspace / "output"
    nested_output = nested_output_parent / "result"
    nested_output_parent.mkdir(parents=True)
    nested_source.write_bytes(stream.read_bytes())
    nested_environment = run._worker_environment()  # noqa: SLF001
    nested_environment["PYTHONPATH"] = os.pathsep.join(
        (os.fspath(captured), os.fspath(captured / "src"))
    )
    nested_environment["COMP009_CAPTURED_SOURCE"] = "1"
    nested_environment["COMP009_CAPTURED_ROOT"] = os.fspath(captured)
    nested_command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        run._CAPTURED_COMP009_SCRIPT,  # noqa: SLF001
        os.fspath(captured),
        os.fspath(nested_source),
        os.fspath(native),
        os.fspath(nested_output),
        comp009_authority["input"]["symbol_sha256"],
        comp009_authority["input"]["decoded_state_sha256"],
    ]
    nested_policy = run._worker_launch_policy(  # noqa: SLF001
        command=nested_command,
        cwd=captured,
        environment=nested_environment,
        read_only=[
            *(
                Path(value)
                for value in run._captured_comp009_system_read_only(  # noqa: SLF001
                    policy["read_only_request"]
                )
            ),
            captured,
            nested_source,
            native,
        ],
        read_write=[nested_output_parent],
        denied_read_probes=[],
        launcher_path=Path(run.sandbox.__file__).resolve(strict=True),
        timeout=3600,
    )
    captured_worker = {
        "isolated_python": True,
        "captured_root": os.fspath(captured),
        "module_origins": {
            "benchmarks.ssp2e_arithmetic": "benchmarks/ssp2e_arithmetic.py",
            "benchmarks.ssp2e_execute": "benchmarks/ssp2e_execute.py",
        },
    }
    captured_attestation = _producer_attestation(
        attestation_template,
        policy=nested_policy,
        nonce="5" * 64,
        parent_sandbox_attestation_sha256=str(
            producing_attestation["attestation_sha256"]
        ),
    )
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    captured = {
        "isolated_python_flags": ["-I", "-B"],
        "script_sha256": hashlib.sha256(
            run._CAPTURED_COMP009_SCRIPT.encode("utf-8")  # noqa: SLF001
        ).hexdigest(),
        "command": [
            Path(sys.executable).name,
            "-I",
            "-B",
            "-c",
            "<captured-root>",
            "<source>",
            "<native>",
            "<output>",
            "<symbol-sha256>",
            "<boundary-sha256>",
        ],
        "source_copy": {
            "bytes": stream.stat().st_size,
            "sha256": hashlib.sha256(stream.read_bytes()).hexdigest(),
            "private_copy_path": os.fspath(nested_source),
            "exclusive_write_before_launch": True,
            "identity_verified_from_same_opened_copy_before_launch": True,
        },
        "returncode": 0,
        "stdout_bytes": 0,
        "stdout_sha256": empty_sha256,
        "stdout_hex": "",
        "stderr_bytes": 0,
        "stderr_sha256": empty_sha256,
        "stderr_hex": "",
        "worker": captured_worker,
        "sandbox_attestation": captured_attestation,
        "launch": _launch(
            attestation=captured_attestation,
            worker=captured_worker,
            seal_field="canonical_worker_sha256",
            nonce="5" * 64,
            policy=nested_policy,
            timeout=3600,
        ),
        "captured_source_manifest_sha256": "c" * 64,
        "captured_source_archive_sha256": run.execute.COMP009_CAPTURED_SOURCE_SHA256,
        "deterministic_execution_core_equal": True,
        "decision_use": False,
    }
    cell = actual.seal_record(
        {
            "schema": "structsplat.comp011.candidate-cell.v1",
            "protocol": run.PROTOCOL,
            "task_sha256": run.TASK_SHA256,
            "state": "complete_candidate_cell",
            "image_id": image_id,
            "bit_tuple": list(bits),
            "relative_path": base.relative_to(outdir).as_posix(),
            "execution": execution,
            "artifacts": artifacts,
            "cold_decodes": cold_decodes,
            "diagnostics": {
                "decision_use": False,
                "producing_worker": {
                    "config": producer_config,
                    "worker": producing_worker,
                    "sandbox_attestation": producing_attestation,
                    "launch": producing_launch,
                    "denied_read_probe_paths": [os.fspath(target)],
                },
                "captured_comp009_reproduction": captured,
            },
            "target_accessed": False,
            "confirmation_payloads_accessed": False,
        },
        "cell_record_sha256",
    )

    monkeypatch.setattr(run.execute, "verify_record", lambda _record: None)
    monkeypatch.setattr(
        run.decode_worker, "verify_candidate_cold_seal", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        run.decode_worker, "stream_component_records", lambda _blob: {"total": {}}
    )
    monkeypatch.setattr(
        run.decode_worker, "component_records_sha256", lambda _records: "6" * 64
    )
    monkeypatch.setattr(run, "_cold_decode_expectations", lambda *_args: ("7" * 64, "8" * 64))
    monkeypatch.setattr(run, "_decode_worker_source_record", lambda: worker_source)
    # This fixture audits the producing and nested COMP-009 policies.  Its
    # intentionally generic cold bundles are not the subject of those tests;
    # exact ordinary cold-worker policy behavior has a dedicated adversarial
    # fixture in test_ssp2v_actual_run_preflight_policy.py.
    monkeypatch.setattr(
        run, "_verify_ordinary_decode_worker_policy", lambda *_args, **_kwargs: None
    )
    preflight = actual.seal_record(
        {
            "native": {
                "arithmetic": native_authority,
                "lloyd": lloyd_authority,
            },
            "operational_upstream_roots": {"comp008": os.fspath(tmp_path)},
        },
        "preflight_binding_sha256",
    )
    monkeypatch.setattr(run, "_read_canonical_object", lambda _path: preflight)
    monkeypatch.setattr(
        run,
        "_authority_copy",
        lambda *_args: {
            "comp008": {"targets": [{"relative_path": target.name}]},
            "comp009": {
                "cells": [comp009_authority],
                "source_manifest_sha256": "c" * 64,
            },
        },
    )
    assert run._verify_candidate_cell(outdir, cell) == cell  # noqa: SLF001
    return outdir, cell


def _coherently_mutate_producer(
    cell: Mapping[str, Any], mutation: Callable[[dict[str, Any], dict[str, Any]], None]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(cell))
    producing = result["diagnostics"]["producing_worker"]
    worker = producing["worker"]
    attestation = producing["sandbox_attestation"]
    mutation(worker, attestation)
    if worker.get("producing_worker_sha256"):
        worker = actual.seal_record(worker, "producing_worker_sha256")
        producing["worker"] = worker
    attestation = _reseal_attestation(attestation)
    producing["sandbox_attestation"] = attestation
    producing["launch"] = _launch(
        attestation=attestation,
        worker=worker,
        seal_field="producing_worker_sha256",
        nonce=producing["launch"]["nonce"],
        policy=producing["config"]["producer_policy"],
        rebind_environment=False,
    )
    return actual.seal_record(result, "cell_record_sha256")


def test_candidate_historical_producer_policy_ignores_captured_source_routing(
    persisted_candidate: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relocated replay root cannot replace the original producer authority."""

    outdir, cell = persisted_candidate
    ordinary_source_root = run.ROOT
    captured_source_root = tmp_path / "captured-active-source"
    captured_source_root.mkdir()
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        run,
        "_ordinary_worker_source_root",
        lambda _outdir: ordinary_source_root,
    )
    monkeypatch.setattr(
        run,
        "_worker_source_root",
        lambda _outdir: captured_source_root,
    )
    monkeypatch.setattr(
        run,
        "_verify_producer_launch_policy",
        lambda **kwargs: observed.update(kwargs),
    )

    assert run._verify_candidate_cell(outdir, cell) == cell  # noqa: SLF001
    assert observed["expected_source_root"] == ordinary_source_root
    assert observed["expected_source_root"] != captured_source_root


def _mutate_command(_worker: dict[str, Any], attestation: dict[str, Any]) -> None:
    attestation["command"].extend(["--unbound-input", "/tmp/foreign-input"])
    attestation["command_sha256"] = hashlib.sha256(
        run._canonical_json(attestation["command"])  # noqa: SLF001
    ).hexdigest()


def _mutate_environment(_worker: dict[str, Any], attestation: dict[str, Any]) -> None:
    attestation["environment_keys"] = sorted(
        {*attestation["environment_keys"], "LD_AUDIT"}
    )
    attestation["environment_sha256_before_attestation_marker"] = "9" * 64


def _mutate_cwd(_worker: dict[str, Any], attestation: dict[str, Any]) -> None:
    attestation["cwd"] = "/tmp"


def _mutate_read_only(_worker: dict[str, Any], attestation: dict[str, Any]) -> None:
    attestation["read_only"] = sorted({*attestation["read_only"], "/tmp"})


def _mutate_read_write(_worker: dict[str, Any], attestation: dict[str, Any]) -> None:
    attestation["read_write"] = sorted({*attestation["read_write"], "/var/tmp"})


def _mutate_launcher(_worker: dict[str, Any], attestation: dict[str, Any]) -> None:
    attestation["launcher_sha256"] = "a" * 64


def _mutate_denied_probes(_worker: dict[str, Any], attestation: dict[str, Any]) -> None:
    attestation["denied_read_probes"] = []


def _mutate_root_into_nested(
    _worker: dict[str, Any], attestation: dict[str, Any]
) -> None:
    attestation["restriction_mode"] = "inherited_nested"
    attestation["parent_sandbox_attestation_sha256"] = "1" * 64
    attestation["socket_probe"] = {
        "basis": "inherited_non_unix_creation_denial",
        "inherited_AF_INET_create": errno.EACCES,
        "inherited_AF_INET6_create": errno.EACCES,
        "TCP_bind": None,
        "TCP_connect": None,
        "AF_INET": errno.EACCES,
        "AF_INET6": errno.EACCES,
        "AF_UNIX_allowed": 1,
    }


def _mutate_worker_execution(worker: dict[str, Any], _attestation: dict[str, Any]) -> None:
    worker["execution"] = {"status": "forged"}


def _mutate_worker_artifact(worker: dict[str, Any], _attestation: dict[str, Any]) -> None:
    worker["artifacts"][0]["sha256"] = "b" * 64


@pytest.mark.parametrize(
    "mutation",
    [
        _mutate_command,
        _mutate_environment,
        _mutate_cwd,
        _mutate_read_only,
        _mutate_read_write,
        _mutate_launcher,
        _mutate_denied_probes,
        _mutate_root_into_nested,
        _mutate_worker_execution,
        _mutate_worker_artifact,
    ],
    ids=[
        "command",
        "environment",
        "cwd",
        "read-only",
        "read-write",
        "launcher",
        "denied-target-probe",
        "unexpected-parent",
        "worker-execution",
        "worker-artifacts",
    ],
)
def test_persisted_candidate_rejects_coherently_resealed_producer_policy_mutations(
    persisted_candidate: tuple[Path, dict[str, Any]],
    mutation: Callable[[dict[str, Any], dict[str, Any]], None],
) -> None:
    outdir, cell = persisted_candidate
    hostile = _coherently_mutate_producer(cell, mutation)
    if mutation is _mutate_denied_probes:
        hostile["diagnostics"]["producing_worker"]["denied_read_probe_paths"] = []
        hostile = actual.seal_record(hostile, "cell_record_sha256")
    with pytest.raises((run.LifecycleError, sandbox.SandboxError)):
        run._verify_candidate_cell(outdir, hostile)  # noqa: SLF001


CapturedMutation = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]], None
]


def _coherently_mutate_captured_comp009(
    cell: Mapping[str, Any], mutation: CapturedMutation
) -> dict[str, Any]:
    result = copy.deepcopy(dict(cell))
    captured = result["diagnostics"]["captured_comp009_reproduction"]
    worker = captured["worker"]
    policy = captured["launch"]["worker_policy"]
    mutation(captured, policy, worker)
    policy = actual.seal_record(policy, "worker_policy_sha256")
    nonce = captured["launch"]["nonce"]
    attestation = _producer_attestation(
        captured["sandbox_attestation"],
        policy=policy,
        nonce=nonce,
    )
    captured["sandbox_attestation"] = attestation
    captured["launch"] = _launch(
        attestation=attestation,
        worker=worker,
        seal_field="canonical_worker_sha256",
        nonce=nonce,
        policy=policy,
        timeout=float(policy["timeout_seconds"]),
    )
    return actual.seal_record(result, "cell_record_sha256")


def _captured_script_substitution(
    _captured: dict[str, Any], policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    policy["command"][4] = "pass"


def _captured_foreign_root(
    _captured: dict[str, Any], policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    policy["command"][5] = "/tmp/foreign-comp009-root"


def _captured_workspace_escape(
    captured: dict[str, Any], policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    policy["command"][6] = "/tmp/comp011-comp009-cell-abcdefgh/source.sspl1"
    captured["source_copy"]["private_copy_path"] = policy["command"][6]


def _captured_native_substitution(
    _captured: dict[str, Any], policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    policy["command"][7] = "/tmp/foreign-arithmetic.so"


def _captured_authority_substitution(
    _captured: dict[str, Any], policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    policy["command"][9] = "f" * 64


def _captured_environment_expansion(
    _captured: dict[str, Any], policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    policy["environment"]["LD_AUDIT"] = "/tmp/foreign-audit.so"


def _captured_read_expansion(
    _captured: dict[str, Any], policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    policy["read_only_request"] = sorted(
        {*policy["read_only_request"], "/tmp/foreign-read"}
    )


def _captured_runtime_read_removal(
    _captured: dict[str, Any], policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    policy["read_only_request"].remove(
        run._worker_policy_path(Path("/dev/null"))  # noqa: SLF001
    )


def _captured_write_expansion(
    _captured: dict[str, Any], policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    policy["read_write_request"] = sorted(
        {*policy["read_write_request"], "/tmp/foreign-write"}
    )


def _captured_launcher_substitution(
    _captured: dict[str, Any], policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    policy["launcher_path"] = "/tmp/foreign-launcher.py"
    policy["launcher_sha256"] = "a" * 64


def _captured_denied_probe_substitution(
    _captured: dict[str, Any], policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    policy["denied_read_probe_paths"] = ["/tmp/foreign-probe"]


def _captured_timeout_substitution(
    _captured: dict[str, Any], policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    policy["timeout_seconds"] = 3599.0


def _captured_worker_source_substitution(
    _captured: dict[str, Any], _policy: dict[str, Any], worker: dict[str, Any]
) -> None:
    worker["module_origins"]["benchmarks.ssp2e_execute"] = "foreign.py"


def _captured_source_receipt_substitution(
    captured: dict[str, Any], _policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    captured["source_copy"]["sha256"] = "b" * 64


def _captured_manifest_substitution(
    captured: dict[str, Any], _policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    captured["captured_source_manifest_sha256"] = "9" * 64


def _captured_parent_substitution(
    captured: dict[str, Any], _policy: dict[str, Any], _worker: dict[str, Any]
) -> None:
    captured["sandbox_attestation"]["parent_sandbox_attestation_sha256"] = "0" * 64


@pytest.mark.parametrize(
    "mutation",
    [
        _captured_script_substitution,
        _captured_foreign_root,
        _captured_workspace_escape,
        _captured_native_substitution,
        _captured_authority_substitution,
        _captured_environment_expansion,
        _captured_read_expansion,
        _captured_runtime_read_removal,
        _captured_write_expansion,
        _captured_launcher_substitution,
        _captured_denied_probe_substitution,
        _captured_timeout_substitution,
        _captured_worker_source_substitution,
        _captured_source_receipt_substitution,
        _captured_manifest_substitution,
        _captured_parent_substitution,
    ],
    ids=[
        "script",
        "captured-root",
        "workspace",
        "native",
        "symbol-authority",
        "environment",
        "read-only",
        "missing-runtime-read",
        "read-write",
        "launcher",
        "denied-probe",
        "timeout",
        "module-origin",
        "source-receipt",
        "source-manifest",
        "parent-lineage",
    ],
)
def test_persisted_candidate_rejects_coherently_resealed_nested_comp009_mutations(
    persisted_candidate: tuple[Path, dict[str, Any]],
    mutation: CapturedMutation,
) -> None:
    outdir, cell = persisted_candidate
    hostile = _coherently_mutate_captured_comp009(cell, mutation)
    with pytest.raises((run.LifecycleError, sandbox.SandboxError)):
        run._verify_candidate_cell(outdir, hostile)  # noqa: SLF001


def test_replay_phase_a_reverifies_nested_comp009_policy() -> None:
    source = inspect.getsource(run._verify_replay_phase_a)  # noqa: SLF001
    assert "_verify_captured_comp009_reproduction_policy(" in source


@pytest.fixture(scope="module")
def ordinary_runtime_policy_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    outdir = tmp_path_factory.mktemp("ordinary-runtime-policy") / "artifact"
    outdir.mkdir()
    manifest = run.source_manifest()
    archive = outdir / "source_snapshot.tar"
    run.write_source_archive(archive, manifest)
    source_root = outdir / run._RUNTIME_SOURCE_RELATIVE  # noqa: SLF001
    run.safe_extract_source_archive(archive, source_root, expected_manifest=manifest)
    blob = outdir / "candidates/image/cell/exact/sspl1.bin"
    target = outdir / "targets/image/target.rgb"
    native = outdir / "runtime/native/arithmetic.so"
    renderer = outdir / "runtime/renderer/record.json"
    foreign_read = outdir / "foreign-read.bin"
    foreign_write = outdir / "foreign-write"
    for path, payload in (
        (blob, b"stream"),
        (target, b"\x00\x00\x00"),
        (native, b"native"),
        (renderer, b"renderer"),
        (foreign_read, b"foreign"),
    ):
        run._exclusive_write(path, payload)  # noqa: SLF001
    foreign_write.mkdir()
    return {
        "outdir": outdir,
        "source_root": source_root,
        "blob": blob,
        "target": target,
        "native": native,
        "renderer": renderer,
        "foreign_read": foreign_read,
        "foreign_write": foreign_write,
    }


def _ordinary_read_only(outdir: Path, source_root: Path, *extra: Path) -> list[Path]:
    return [
        *run._sandbox_system_read_only(),  # noqa: SLF001
        *run._worker_source_read_only(source_root),  # noqa: SLF001
        outdir / "runtime",
        *extra,
    ]


def _ordinary_policy(
    *,
    outdir: Path,
    source_root: Path,
    command: list[str],
    read_only: list[Path],
    read_write: list[Path] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return run._worker_launch_policy(  # noqa: SLF001
        command=command,
        cwd=source_root / "benchmarks",
        environment=dict(environment or run._worker_environment(outdir)),  # noqa: SLF001
        read_only=read_only,
        read_write=list(read_write or []),
        denied_read_probes=[],
        launcher_path=source_root / "benchmarks/ssp2v_landlock.py",
        timeout=600,
    )


def test_ordinary_decode_policy_rejects_coherently_resealed_capability_drift(
    ordinary_runtime_policy_artifact: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = ordinary_runtime_policy_artifact
    outdir = fixture["outdir"]
    source_root = fixture["source_root"]
    blob = fixture["blob"]
    command = [sys.executable, "-B", "-c", "print('sealed decode')"]
    policy = _ordinary_policy(
        outdir=outdir,
        source_root=source_root,
        command=command,
        read_only=_ordinary_read_only(outdir, source_root, blob),
    )
    bundle = {"launch": {"worker_policy": policy}}
    run._verify_ordinary_decode_worker_policy(  # noqa: SLF001
        bundle, outdir=outdir, expected_command=command, blob_path=blob
    )
    assert policy["cwd"] == os.fspath(source_root / "benchmarks")
    assert policy["environment"]["PYTHONPATH"].split(os.pathsep)[1:] == [
        os.fspath(source_root),
        os.fspath(source_root / "src"),
    ]
    original_environment = run._worker_environment  # noqa: SLF001
    explicit_roots: list[Path | None] = []

    def observe_source_root(
        root: Path | None = None, *, source_root: Path | None = None
    ) -> dict[str, str]:
        explicit_roots.append(source_root)
        return original_environment(root, source_root=source_root)

    monkeypatch.setattr(run, "_worker_environment", observe_source_root)
    run._verify_ordinary_decode_worker_policy(  # noqa: SLF001
        bundle, outdir=outdir, expected_command=command, blob_path=blob
    )
    assert explicit_roots and set(explicit_roots) == {source_root}

    def mutate(name: str, value: dict[str, Any]) -> None:
        if name == "command":
            value["command"][-1] = "print('foreign')"
        elif name == "cwd":
            value["cwd"] = os.fspath(source_root / "tests")
        elif name == "environment":
            value["environment"]["LD_AUDIT"] = "/tmp/foreign-audit.so"
        elif name == "read-only":
            value["read_only_request"] = sorted(
                {*value["read_only_request"], os.fspath(fixture["foreign_read"])}
            )
        elif name == "read-write":
            value["read_write_request"] = [os.fspath(fixture["foreign_write"])]
        elif name == "denied":
            value["denied_read_probe_paths"] = [os.fspath(fixture["foreign_read"])]
        elif name == "launcher":
            live = Path(sandbox.__file__).resolve(strict=True)
            value["launcher_path"] = os.fspath(live)
            value["launcher_sha256"] = hashlib.sha256(live.read_bytes()).hexdigest()
        else:
            value["timeout_seconds"] = 601.0

    for name in (
        "command",
        "cwd",
        "environment",
        "read-only",
        "read-write",
        "denied",
        "launcher",
        "timeout",
    ):
        hostile = copy.deepcopy(policy)
        mutate(name, hostile)
        hostile = actual.seal_record(hostile, "worker_policy_sha256")
        with pytest.raises(run.LifecycleError, match="exact frozen policy"):
            run._verify_ordinary_decode_worker_policy(  # noqa: SLF001
                {"launch": {"worker_policy": hostile}},
                outdir=outdir,
                expected_command=command,
                blob_path=blob,
            )


def test_ordinary_quality_policies_reject_coherently_resealed_capability_drift(
    ordinary_runtime_policy_artifact: Mapping[str, Any],
) -> None:
    fixture = ordinary_runtime_policy_artifact
    outdir = fixture["outdir"]
    source_root = fixture["source_root"]
    blob = fixture["blob"]
    target = fixture["target"]
    native = fixture["native"]
    renderer = fixture["renderer"]
    render_temp = outdir / "comp011-render-metric-abcdefgh"
    render_temp.mkdir()
    render_path = render_temp / "render.npy"
    run._exclusive_write(render_path, b"render")  # noqa: SLF001
    expected_stream = {
        "blob_sha256": hashlib.sha256(blob.read_bytes()).hexdigest(),
        "symbol_sha256": "1" * 64,
        "boundary_state_sha256": "2" * 64,
    }
    native_record = {
        "path": native.name,
        "bytes": native.stat().st_size,
        "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
    }
    render_record = {
        "render_npy": {"sha256": hashlib.sha256(render_path.read_bytes()).hexdigest()},
        "render_tensor": {"sha256": "3" * 64},
    }
    target_record = {
        "rgb_copy": {
            "relative_path": target.relative_to(outdir).as_posix(),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        "rgb_dimensions_hw": [1, 1],
    }
    lpips = "4" * 64
    render_command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_actual_run",
        "_render-arm",
        "--artifact-root",
        os.fspath(outdir),
        "--renderer-record",
        os.fspath(renderer),
        "--blob",
        os.fspath(blob),
        "--native",
        os.fspath(native),
        "--expected-blob-sha256",
        expected_stream["blob_sha256"],
        "--expected-native-sha256",
        native_record["sha256"],
        "--expected-native-bytes",
        str(native_record["bytes"]),
        "--expected-symbol-sha256",
        expected_stream["symbol_sha256"],
        "--expected-boundary-state-sha256",
        expected_stream["boundary_state_sha256"],
        "--output",
        os.fspath(render_path),
    ]
    render_environment = run._worker_environment(outdir)  # noqa: SLF001
    render_environment["TMPDIR"] = os.fspath(render_temp)
    render_policy = _ordinary_policy(
        outdir=outdir,
        source_root=source_root,
        command=render_command,
        environment=render_environment,
        read_only=_ordinary_read_only(outdir, source_root, blob),
        read_write=[render_temp, *run._sandbox_gpu_read_write()],  # noqa: SLF001
    )
    metric_command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_actual_run",
        "_metric-worker",
        "--render",
        os.fspath(render_path),
        "--target-rgb",
        os.fspath(target),
        "--height",
        "1",
        "--width",
        "1",
        "--expected-render-npy-sha256",
        render_record["render_npy"]["sha256"],
        "--expected-render-tensor-sha256",
        render_record["render_tensor"]["sha256"],
        "--expected-target-rgb-sha256",
        target_record["rgb_copy"]["sha256"],
        "--expected-lpips-state-sha256",
        lpips,
    ]
    metric_scratch = render_temp / "metric-scratch"
    metric_scratch.mkdir()
    metric_environment = run._worker_environment(outdir)  # noqa: SLF001
    metric_environment["TMPDIR"] = os.fspath(metric_scratch)
    metric_policy = _ordinary_policy(
        outdir=outdir,
        source_root=source_root,
        command=metric_command,
        environment=metric_environment,
        read_only=_ordinary_read_only(outdir, source_root, render_path, target),
        read_write=[metric_scratch],
    )
    render_path.unlink()
    metric_scratch.rmdir()
    render_temp.rmdir()
    task = {
        "render_launch": {"worker_policy": render_policy},
        "metric_launch": {"worker_policy": metric_policy},
    }
    preflight = {"native": {"arithmetic": native_record}}
    run._verify_ordinary_render_metric_policy(  # noqa: SLF001
        task,
        outdir=outdir,
        blob_path=blob,
        expected_stream=expected_stream,
        render=render_record,
        target_record=target_record,
        preflight_record=preflight,
        expected_lpips_state_sha256=lpips,
    )

    mutations = (
        ("render_launch", "command"),
        ("render_launch", "cwd"),
        ("render_launch", "environment"),
        ("render_launch", "read_only_request"),
        ("render_launch", "read_write_request"),
        ("render_launch", "denied_read_probe_paths"),
        ("render_launch", "launcher_sha256"),
        ("render_launch", "timeout_seconds"),
        ("metric_launch", "command"),
        ("metric_launch", "environment"),
        ("metric_launch", "read_only_request"),
        ("metric_launch", "read_write_request"),
        ("metric_launch", "launcher_sha256"),
        ("metric_launch", "timeout_seconds"),
    )
    for launch_name, field in mutations:
        hostile = copy.deepcopy(task)
        policy = hostile[launch_name]["worker_policy"]
        if field == "command":
            policy["command"][3] = "_import-stream"
        elif field == "cwd":
            policy[field] = os.fspath(source_root / "tests")
        elif field == "environment":
            policy[field]["LD_AUDIT"] = "/tmp/foreign-audit.so"
        elif field in {"read_only_request", "read_write_request"}:
            policy[field] = sorted(
                {*policy[field], os.fspath(fixture["foreign_read"])}
            )
        elif field == "denied_read_probe_paths":
            policy[field] = [os.fspath(fixture["foreign_read"])]
        elif field == "launcher_sha256":
            policy[field] = "0" * 64
        else:
            policy[field] = 601.0
        hostile[launch_name]["worker_policy"] = actual.seal_record(
            policy, "worker_policy_sha256"
        )
        with pytest.raises(run.LifecycleError):
            run._verify_ordinary_render_metric_policy(  # noqa: SLF001
                hostile,
                outdir=outdir,
                blob_path=blob,
                expected_stream=expected_stream,
                render=render_record,
                target_record=target_record,
                preflight_record=preflight,
                expected_lpips_state_sha256=lpips,
            )


def test_ordinary_resource_command_is_bound_by_exact_decode_policy(
    ordinary_runtime_policy_artifact: Mapping[str, Any],
) -> None:
    fixture = ordinary_runtime_policy_artifact
    outdir = fixture["outdir"]
    source_root = fixture["source_root"]
    blob = fixture["blob"]
    native = fixture["native"]
    stream = {
        "blob_bytes": blob.stat().st_size,
        "blob_sha256": hashlib.sha256(blob.read_bytes()).hexdigest(),
        "symbol_sha256": "5" * 64,
        "boundary_state_sha256": "6" * 64,
    }
    native_record = {
        "bytes": native.stat().st_size,
        "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
    }
    worker = run._decode_worker_source_record()  # noqa: SLF001
    launch_spec = run.decode_worker.LaunchSpec("secondary", 0, "warmup", None, "sspl1")
    command = run._resource_worker_command(  # noqa: SLF001
        artifact_root=outdir,
        blob_relative_path=blob.relative_to(outdir).as_posix(),
        native_relative_path=native.relative_to(outdir).as_posix(),
        stream_record=stream,
        native_record=native_record,
        worker_source=worker,
        measurement_session_sha256="7" * 64,
        launch=launch_spec,
    )
    policy = _ordinary_policy(
        outdir=outdir,
        source_root=source_root,
        command=command,
        read_only=_ordinary_read_only(outdir, source_root, blob),
    )
    run._verify_ordinary_decode_worker_policy(  # noqa: SLF001
        {"launch": {"worker_policy": policy}},
        outdir=outdir,
        expected_command=command,
        blob_path=blob,
    )
    hostile = copy.deepcopy(policy)
    hostile["command"][hostile["command"].index("--arm") + 1] = "ssp2l"
    hostile = actual.seal_record(hostile, "worker_policy_sha256")
    with pytest.raises(run.LifecycleError, match="exact frozen policy"):
        run._verify_ordinary_decode_worker_policy(  # noqa: SLF001
            {"launch": {"worker_policy": hostile}},
            outdir=outdir,
            expected_command=command,
            blob_path=blob,
        )
