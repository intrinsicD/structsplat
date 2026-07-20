from __future__ import annotations

import errno
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from benchmarks import ssp2v_landlock as sandbox


_SYSTEM_READ = [Path(sys.prefix), Path("/usr")]
_ROOT = Path(__file__).resolve(strict=True).parents[1]
_CLEAN_ENV = {
    "HOME": "/nonexistent-structsplat-sandbox-home",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": os.environ["PATH"],
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
_NESTED_ENV = {**_CLEAN_ENV, "PYTHONPATH": os.fspath(_ROOT)}


def _nested_reads(*paths: Path) -> list[Path]:
    return [*_SYSTEM_READ, Path("/dev/null"), _ROOT, *paths]


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _process_is_absent(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _expected_launch_lineage() -> tuple[str, str | None]:
    parent = os.environ.get(sandbox.SANDBOX_ATTESTATION_ENV)
    return (
        (
            sandbox.RESTRICTION_MODE_FRESH
            if parent is None
            else sandbox.RESTRICTION_MODE_INHERITED
        ),
        parent,
    )


def test_landlock_abi_supports_frozen_filesystem_network_and_ipc_rules() -> None:
    assert sandbox.landlock_abi() >= 6


def test_run_sandboxed_enforces_filesystem_and_socket_boundaries(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    writable = tmp_path / "writable"
    denied = tmp_path / "denied"
    for path in (allowed, writable, denied):
        path.mkdir()
    (allowed / "input.txt").write_text("allowed", encoding="utf-8")
    (denied / "secret.txt").write_text("secret", encoding="utf-8")

    pathname_socket_path = tmp_path / "outside.sock"
    abstract_socket_name = f"structsplat-comp011-{os.getpid()}-{tmp_path.name}"
    pathname_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    abstract_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    pathname_server.bind(os.fspath(pathname_socket_path))
    abstract_server.bind("\0" + abstract_socket_name)
    pathname_server.listen()
    abstract_server.listen()

    script = allowed / "worker.py"
    script.write_text(
        """
import fcntl
import json
import os
import socket
import sys
from pathlib import Path

allowed, writable, denied = map(Path, sys.argv[1:4])
pathname_socket, abstract_socket_name, proc_status_expectation = sys.argv[4:]
assert (allowed / "input.txt").read_text() == "allowed"
try:
    proc_status = Path("/proc/self/status").read_text()
except PermissionError:
    proc_status_allowed = False
else:
    proc_status_allowed = proc_status.startswith("Name:")
assert proc_status_allowed is (proc_status_expectation == "allowed")
(writable / "output.txt").write_text("written")
with Path("/dev/null").open("wb") as handle:
    assert handle.write(b"device-probe") == 12
try:
    (denied / "secret.txt").read_bytes()
except PermissionError:
    denied_read = True
else:
    denied_read = False
families = {}
for family, name in ((socket.AF_INET, "inet"), (socket.AF_INET6, "inet6")):
    try:
        socket.socket(family, socket.SOCK_DGRAM)
    except PermissionError:
        families[name] = "denied"
    else:
        families[name] = "allowed"
families["unix"] = "allowed"
value = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
value.close()
ipc = {}
for address, name in (
    (pathname_socket, "pathname_connect"),
    ("\\0" + abstract_socket_name, "abstract_connect"),
):
    value = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        value.connect(address)
    except PermissionError as error:
        ipc[name] = error.errno
    else:
        ipc[name] = "allowed"
    finally:
        value.close()
try:
    os.kill(os.getppid(), 0)
except PermissionError as error:
    ipc["parent_signal"] = error.errno
else:
    ipc["parent_signal"] = "allowed"
local_pair = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
for value in local_pair:
    value.close()
ipc["private_unix_socketpair"] = "allowed"
value = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
value.bind(str(writable / "private.sock"))
value.close()
ipc["private_pathname_bind"] = "allowed"
value = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
value.bind("\\0private-" + abstract_socket_name)
value.close()
ipc["private_abstract_bind"] = "allowed"
leaked_fds = []
for descriptor in range(3, 64):
    try:
        fcntl.fcntl(descriptor, fcntl.F_GETFD)
    except OSError:
        pass
    else:
        leaked_fds.append(descriptor)
print(json.dumps({
    "denied_read": denied_read,
    "environment_keys": sorted(os.environ),
    "families": families,
    "ipc": ipc,
    "leaked_fds": leaked_fds,
    "sandbox_sha256": os.environ["STRUCTSPLAT_SANDBOX_ATTESTATION_SHA256"],
}, sort_keys=True))
""".lstrip(),
        encoding="utf-8",
    )
    expected_mode, expected_parent = _expected_launch_lineage()
    completed, attestation = sandbox.run_sandboxed(
        [
            sys.executable,
            "-I",
            "-B",
            os.fspath(script),
            os.fspath(allowed),
            os.fspath(writable),
            os.fspath(denied),
            os.fspath(pathname_socket_path),
            abstract_socket_name,
            (
                "allowed"
                if expected_mode == sandbox.RESTRICTION_MODE_FRESH
                else "denied"
            ),
        ],
        read_only=[*_SYSTEM_READ, Path("/proc"), allowed],
        read_write=[Path("/dev/null"), writable],
        denied_read_probes=[denied / "secret.txt"],
        cwd=allowed,
        env=_CLEAN_ENV,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    output = json.loads(completed.stdout)
    assert output == {
        "denied_read": True,
        "environment_keys": sorted([*_CLEAN_ENV, "STRUCTSPLAT_SANDBOX_ATTESTATION_SHA256"]),
        "families": {"inet": "denied", "inet6": "denied", "unix": "allowed"},
        "ipc": {
            "abstract_connect": 13,
            # The seccomp filter deliberately permits kill(2) so a confined
            # parent can reap descendants.  Landlock signal scoping still
            # denies this signal to the ancestor domain.
            "parent_signal": errno.EPERM,
            "private_abstract_bind": "allowed",
            "private_pathname_bind": "allowed",
            "pathname_connect": 13,
            "private_unix_socketpair": "allowed",
        },
        "leaked_fds": [],
        "sandbox_sha256": attestation["attestation_sha256"],
    }
    assert (writable / "output.txt").read_text(encoding="utf-8") == "written"
    assert attestation["filesystem_restricted"] is True
    assert attestation["restriction_mode"] == expected_mode
    assert attestation["parent_sandbox_attestation_sha256"] == expected_parent
    assert attestation["parent_death_signal"] == signal.SIGKILL
    assert attestation["parent_death_signal_verified"] is True
    assert attestation["parent_pid_race_checked"] is True
    if expected_mode == sandbox.RESTRICTION_MODE_FRESH:
        expected_socket_prefix = {
            "basis": "direct_bind_connect",
            "inherited_AF_INET_create": None,
            "inherited_AF_INET6_create": None,
            "TCP_bind": errno.EACCES,
            "TCP_connect": errno.EACCES,
        }
    else:
        expected_socket_prefix = {
            "basis": "inherited_non_unix_creation_denial",
            "inherited_AF_INET_create": errno.EACCES,
            "inherited_AF_INET6_create": errno.EACCES,
            "TCP_bind": None,
            "TCP_connect": None,
        }
    assert attestation["socket_probe"] == {
        **expected_socket_prefix,
        "AF_INET": errno.EACCES,
        "AF_INET6": errno.EACCES,
        "AF_UNIX_allowed": 1,
    }
    assert attestation["tcp_bind_connect_restricted"] is True
    assert attestation["non_unix_socket_creation_restricted"] is True
    assert attestation["command"][-6:-1] == [
        os.fspath(allowed),
        os.fspath(writable),
        os.fspath(denied),
        os.fspath(pathname_socket_path),
        abstract_socket_name,
    ]
    assert attestation["cwd"] == os.fspath(allowed)
    assert attestation["environment_keys"] == sorted(_CLEAN_ENV)
    assert attestation["external_unix_ipc_restricted"] is True
    assert attestation["cross_process_restricted"] is True
    assert attestation["landlock_scope"] == ["abstract_unix_socket", "signal"]
    assert attestation["denied_read_probes"] == [
        {"errno": 13, "path": os.fspath((denied / "secret.txt").resolve())}
    ]
    sandbox.verify_attestation(attestation)
    pathname_server.close()
    abstract_server.close()


def test_run_sandboxed_fails_closed_without_readable_command(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    script.write_text("print('must not run')\n", encoding="utf-8")
    completed, attestation = sandbox.run_sandboxed(
        [sys.executable, "-I", "-B", os.fspath(script)],
        read_only=_SYSTEM_READ,
        read_write=[],
        cwd=tmp_path,
        env=_CLEAN_ENV,
        timeout=30,
    )
    assert completed.returncode != 0
    assert completed.stdout == b""
    assert attestation["filesystem_restricted"] is True


def test_attestation_rejects_tampering(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    script = allowed / "worker.py"
    script.write_text("pass\n", encoding="utf-8")
    completed, attestation = sandbox.run_sandboxed(
        [sys.executable, "-I", "-B", os.fspath(script)],
        read_only=[*_SYSTEM_READ, allowed],
        read_write=[],
        cwd=allowed,
        env=_CLEAN_ENV,
        timeout=30,
    )
    assert completed.returncode == 0
    tampered = dict(attestation)
    tampered["filesystem_restricted"] = False
    with pytest.raises(sandbox.SandboxError, match="seal mismatch"):
        sandbox.verify_attestation(tampered)


def test_rejects_relative_missing_and_overlapping_allowlists(tmp_path: Path) -> None:
    with pytest.raises(sandbox.SandboxError, match="must be absolute"):
        sandbox.run_sandboxed(
            [sys.executable, "-c", "pass"],
            read_only=[Path("relative")],
            read_write=[],
            env=_CLEAN_ENV,
            timeout=30,
        )
    missing = tmp_path / "missing"
    with pytest.raises(sandbox.SandboxError, match="is not an existing path"):
        sandbox.run_sandboxed(
            [sys.executable, "-c", "pass"],
            read_only=[missing],
            read_write=[],
            env=_CLEAN_ENV,
            timeout=30,
        )
    with pytest.raises(sandbox.SandboxError, match="did not emit one valid attestation"):
        sandbox.run_sandboxed(
            [sys.executable, "-c", "pass"],
            read_only=[tmp_path],
            read_write=[tmp_path],
            env=_CLEAN_ENV,
            timeout=30,
        )


def test_nested_policy_stacks_with_lineage_and_cannot_promote_parent_authority(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    visible = tmp_path / "visible"
    outer_only = tmp_path / "outer-only"
    inherited_readonly = tmp_path / "inherited-readonly"
    for path in (scripts, visible, outer_only, inherited_readonly):
        path.mkdir()
    (visible / "input.txt").write_text("visible", encoding="utf-8")
    (outer_only / "secret.txt").write_text("outer-only", encoding="utf-8")
    (inherited_readonly / "locked.txt").write_text("locked", encoding="utf-8")

    inner = scripts / "inner.py"
    inner.write_text(
        """
import json
import sys
from pathlib import Path

visible, outer_only, inherited_readonly = map(Path, sys.argv[1:])
visible_value = (visible / "input.txt").read_text(encoding="utf-8")
try:
    (outer_only / "secret.txt").read_text(encoding="utf-8")
except PermissionError:
    narrower_read_denied = True
else:
    narrower_read_denied = False
try:
    (inherited_readonly / "locked.txt").write_text("promoted", encoding="utf-8")
except PermissionError:
    inherited_readonly_stayed_readonly = True
else:
    inherited_readonly_stayed_readonly = False
print(json.dumps({
    "inherited_readonly_stayed_readonly": inherited_readonly_stayed_readonly,
    "narrower_read_denied": narrower_read_denied,
    "visible_value": visible_value,
}, sort_keys=True))
""".lstrip(),
        encoding="utf-8",
    )
    outer = scripts / "outer.py"
    outer.write_text(
        """
import json
import os
import sys
from pathlib import Path

from benchmarks import ssp2v_landlock as sandbox

inner, visible, outer_only, inherited_readonly, root = map(Path, sys.argv[1:])
environment = {
    key: value
    for key, value in os.environ.items()
    if key != sandbox.SANDBOX_ATTESTATION_ENV
}
completed, attestation = sandbox.run_sandboxed(
    [
        sys.executable,
        "-B",
        str(inner),
        str(visible),
        str(outer_only),
        str(inherited_readonly),
    ],
    read_only=[
        Path(sys.prefix),
        Path("/usr"),
        Path("/dev/null"),
        root,
        inner.parent,
        visible,
    ],
    # The nested rule requests write access, but the parent granted this path
    # read-only.  Stacked Landlock domains must not permit promotion.
    read_write=[inherited_readonly],
    cwd=inner.parent,
    env=environment,
    timeout=30,
)
if completed.returncode != 0:
    raise RuntimeError(completed.stderr.decode(errors="replace"))
print(json.dumps({
    "inner_attestation": attestation,
    "inner_output": json.loads(completed.stdout),
    "outer_digest": os.environ[sandbox.SANDBOX_ATTESTATION_ENV],
}, sort_keys=True))
""".lstrip(),
        encoding="utf-8",
    )

    completed, outer_attestation = sandbox.run_sandboxed(
        [
            sys.executable,
            "-B",
            os.fspath(outer),
            os.fspath(inner),
            os.fspath(visible),
            os.fspath(outer_only),
            os.fspath(inherited_readonly),
            os.fspath(_ROOT),
        ],
        read_only=_nested_reads(
            scripts, visible, outer_only, inherited_readonly
        ),
        read_write=[],
        cwd=scripts,
        env=_NESTED_ENV,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    payload = json.loads(completed.stdout)
    inner_attestation = payload["inner_attestation"]
    expected_mode, expected_parent = _expected_launch_lineage()
    assert outer_attestation["restriction_mode"] == expected_mode
    assert (
        outer_attestation["parent_sandbox_attestation_sha256"] == expected_parent
    )
    assert payload["outer_digest"] == outer_attestation["attestation_sha256"]
    assert inner_attestation["restriction_mode"] == sandbox.RESTRICTION_MODE_INHERITED
    assert (
        inner_attestation["parent_sandbox_attestation_sha256"]
        == outer_attestation["attestation_sha256"]
    )
    assert inner_attestation["socket_probe"] == {
        "basis": "inherited_non_unix_creation_denial",
        "inherited_AF_INET_create": errno.EACCES,
        "inherited_AF_INET6_create": errno.EACCES,
        "TCP_bind": None,
        "TCP_connect": None,
        "AF_INET": errno.EACCES,
        "AF_INET6": errno.EACCES,
        "AF_UNIX_allowed": 1,
    }
    assert payload["inner_output"] == {
        "inherited_readonly_stayed_readonly": True,
        "narrower_read_denied": True,
        "visible_value": "visible",
    }
    assert (inherited_readonly / "locked.txt").read_text(encoding="utf-8") == "locked"


def test_three_level_nested_lineage_chain(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    driver = scripts / "driver.py"
    driver.write_text(
        """
import json
import os
import sys
from pathlib import Path

from benchmarks import ssp2v_landlock as sandbox

depth = int(sys.argv[1])
root = Path(sys.argv[2])
scripts = Path(sys.argv[3])
self_digest = os.environ[sandbox.SANDBOX_ATTESTATION_ENV]
if depth == 0:
    print(json.dumps({"self_digest": self_digest}, sort_keys=True))
    raise SystemExit(0)
environment = {
    key: value
    for key, value in os.environ.items()
    if key != sandbox.SANDBOX_ATTESTATION_ENV
}
completed, attestation = sandbox.run_sandboxed(
    [sys.executable, "-B", str(Path(__file__)), str(depth - 1), str(root), str(scripts)],
    read_only=[
        Path(sys.prefix),
        Path("/usr"),
        Path("/dev/null"),
        root,
        scripts,
    ],
    read_write=[],
    cwd=scripts,
    env=environment,
    timeout=30,
)
if completed.returncode != 0:
    raise RuntimeError(completed.stderr.decode(errors="replace"))
print(json.dumps({
    "child": json.loads(completed.stdout),
    "child_attestation": attestation,
    "self_digest": self_digest,
}, sort_keys=True))
""".lstrip(),
        encoding="utf-8",
    )
    completed, first = sandbox.run_sandboxed(
        [
            sys.executable,
            "-B",
            os.fspath(driver),
            "2",
            os.fspath(_ROOT),
            os.fspath(scripts),
        ],
        read_only=_nested_reads(scripts),
        read_write=[],
        cwd=scripts,
        env=_NESTED_ENV,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    first_output = json.loads(completed.stdout)
    second = first_output["child_attestation"]
    second_output = first_output["child"]
    third = second_output["child_attestation"]
    third_output = second_output["child"]
    assert first_output["self_digest"] == first["attestation_sha256"]
    assert second["restriction_mode"] == sandbox.RESTRICTION_MODE_INHERITED
    assert (
        second["parent_sandbox_attestation_sha256"]
        == first["attestation_sha256"]
    )
    assert second_output["self_digest"] == second["attestation_sha256"]
    assert third["restriction_mode"] == sandbox.RESTRICTION_MODE_INHERITED
    assert (
        third["parent_sandbox_attestation_sha256"]
        == second["attestation_sha256"]
    )
    assert third_output == {"self_digest": third["attestation_sha256"]}


def test_reserved_child_marker_and_forged_caller_lineage_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = [sys.executable, "-I", "-B", "-c", "pass"]
    with pytest.raises(sandbox.SandboxError, match="reserved attestation marker"):
        sandbox.run_sandboxed(
            command,
            read_only=_SYSTEM_READ,
            read_write=[],
            env={
                **_CLEAN_ENV,
                sandbox.SANDBOX_ATTESTATION_ENV: "a" * 64,
            },
            timeout=30,
        )

    # A digest-shaped marker alone cannot claim nested status: the launcher
    # also requires an inherited seccomp filter and real AF_INET/AF_INET6
    # creation denials.
    if sandbox.SANDBOX_ATTESTATION_ENV in os.environ:
        # The focused proof intentionally runs this file inside a real parent
        # domain.  The independent outer regression exercises forged-root
        # rejection; persisted parent/child digest equality audits mutations
        # within an already restricted domain.
        return
    monkeypatch.setenv(sandbox.SANDBOX_ATTESTATION_ENV, "b" * 64)
    with pytest.raises(sandbox.SandboxError, match="did not emit one valid attestation"):
        sandbox.run_sandboxed(
            command,
            read_only=[*_SYSTEM_READ, Path("/dev/null")],
            read_write=[],
            cwd=tmp_path,
            env=_CLEAN_ENV,
            timeout=30,
        )


def test_missing_nested_lineage_marker_fails_closed(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    inner = scripts / "inner.py"
    inner.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
    outer = scripts / "outer.py"
    outer.write_text(
        """
import json
import os
import sys
from pathlib import Path

from benchmarks import ssp2v_landlock as sandbox

inner, root = map(Path, sys.argv[1:])
os.environ.pop(sandbox.SANDBOX_ATTESTATION_ENV)
try:
    sandbox.run_sandboxed(
        [sys.executable, "-B", str(inner)],
        read_only=[
            Path(sys.prefix),
            Path("/usr"),
            Path("/dev/null"),
            root,
            inner.parent,
        ],
        read_write=[],
        cwd=inner.parent,
        env={
            key: value
            for key, value in os.environ.items()
            if key != sandbox.SANDBOX_ATTESTATION_ENV
        },
        timeout=30,
    )
except sandbox.SandboxError as exc:
    print(json.dumps({"failed_closed": True, "message": str(exc)}, sort_keys=True))
else:
    raise RuntimeError("missing nested lineage unexpectedly launched a child")
""".lstrip(),
        encoding="utf-8",
    )
    completed, attestation = sandbox.run_sandboxed(
        [
            sys.executable,
            "-B",
            os.fspath(outer),
            os.fspath(inner),
            os.fspath(_ROOT),
        ],
        read_only=_nested_reads(scripts),
        read_write=[],
        cwd=scripts,
        env=_NESTED_ENV,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    expected_mode, expected_parent = _expected_launch_lineage()
    assert attestation["restriction_mode"] == expected_mode
    assert attestation["parent_sandbox_attestation_sha256"] == expected_parent
    output = json.loads(completed.stdout)
    assert output["failed_closed"] is True
    assert "did not emit one valid attestation" in output["message"]


def test_direct_timeout_kills_and_reaps_child(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    state = tmp_path / "state"
    scripts.mkdir()
    state.mkdir()
    sleeper = scripts / "sleeper.py"
    sleeper.write_text(
        """
import os
import sys
import time
from pathlib import Path

Path(sys.argv[1]).write_text(str(os.getpid()), encoding="ascii")
time.sleep(120)
""".lstrip(),
        encoding="utf-8",
    )
    pid_path = state / "direct.pid"
    with pytest.raises(subprocess.TimeoutExpired):
        sandbox.run_sandboxed(
            [sys.executable, "-I", "-B", os.fspath(sleeper), os.fspath(pid_path)],
            read_only=[*_SYSTEM_READ, scripts],
            read_write=[state],
            cwd=scripts,
            env=_CLEAN_ENV,
            timeout=0.5,
        )
    assert pid_path.is_file()
    child_pid = int(pid_path.read_text(encoding="ascii"))
    assert _wait_until(lambda: _process_is_absent(child_pid))


def test_nested_timeout_can_kill_and_reap_direct_descendant(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    state = tmp_path / "state"
    scripts.mkdir()
    state.mkdir()
    sleeper = scripts / "nested-sleeper.py"
    sleeper.write_text(
        """
import os
import sys
import time
from pathlib import Path

Path(sys.argv[1]).write_text(str(os.getpid()), encoding="ascii")
time.sleep(120)
""".lstrip(),
        encoding="utf-8",
    )
    outer = scripts / "timeout-parent.py"
    outer.write_text(
        """
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks import ssp2v_landlock as sandbox

sleeper, pid_path, root = map(Path, sys.argv[1:])
environment = {
    key: value
    for key, value in os.environ.items()
    if key != sandbox.SANDBOX_ATTESTATION_ENV
}
try:
    sandbox.run_sandboxed(
        [sys.executable, "-I", "-B", str(sleeper), str(pid_path)],
        read_only=[
            Path(sys.prefix),
            Path("/usr"),
            Path("/dev/null"),
            root,
            sleeper.parent,
        ],
        read_write=[pid_path.parent],
        cwd=sleeper.parent,
        env=environment,
        timeout=0.5,
    )
except subprocess.TimeoutExpired:
    child_pid = int(pid_path.read_text(encoding="ascii"))
    try:
        os.kill(child_pid, 0)
    except ProcessLookupError:
        child_reaped = True
    else:
        child_reaped = False
    print(json.dumps({
        "child_pid": child_pid,
        "child_reaped": child_reaped,
        "parent_digest": os.environ[sandbox.SANDBOX_ATTESTATION_ENV],
    }, sort_keys=True))
else:
    raise RuntimeError("nested timeout unexpectedly completed")
""".lstrip(),
        encoding="utf-8",
    )
    pid_path = state / "nested.pid"
    completed, attestation = sandbox.run_sandboxed(
        [
            sys.executable,
            "-B",
            os.fspath(outer),
            os.fspath(sleeper),
            os.fspath(pid_path),
            os.fspath(_ROOT),
        ],
        read_only=_nested_reads(scripts),
        read_write=[state],
        cwd=scripts,
        env=_NESTED_ENV,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    output = json.loads(completed.stdout)
    assert output["parent_digest"] == attestation["attestation_sha256"]
    assert output["child_reaped"] is True
    assert _process_is_absent(int(output["child_pid"]))


def test_parent_death_signal_cascades_across_nested_launchers(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    state = tmp_path / "state"
    scripts.mkdir()
    state.mkdir()
    leaf = scripts / "leaf.py"
    leaf.write_text(
        """
import os
import sys
import time
from pathlib import Path

Path(sys.argv[1]).write_text(str(os.getpid()), encoding="ascii")
time.sleep(120)
""".lstrip(),
        encoding="utf-8",
    )
    middle = scripts / "middle.py"
    middle.write_text(
        """
import os
import sys
from pathlib import Path

from benchmarks import ssp2v_landlock as sandbox

leaf, state, root, scripts = map(Path, sys.argv[1:])
(state / "middle.pid").write_text(str(os.getpid()), encoding="ascii")
environment = {
    key: value
    for key, value in os.environ.items()
    if key != sandbox.SANDBOX_ATTESTATION_ENV
}
sandbox.run_sandboxed(
    [sys.executable, "-I", "-B", str(leaf), str(state / "leaf.pid")],
    read_only=[
        Path(sys.prefix),
        Path("/usr"),
        Path("/dev/null"),
        root,
        scripts,
    ],
    read_write=[state],
    cwd=scripts,
    env=environment,
    timeout=120,
)
raise RuntimeError("leaf unexpectedly exited before parent-death test")
""".lstrip(),
        encoding="utf-8",
    )
    controller = scripts / "controller.py"
    controller.write_text(
        """
import sys
from pathlib import Path

from benchmarks import ssp2v_landlock as sandbox

middle, leaf, state, root, scripts = map(Path, sys.argv[1:])
sandbox.run_sandboxed(
    [
        sys.executable,
        "-B",
        str(middle),
        str(leaf),
        str(state),
        str(root),
        str(scripts),
    ],
    read_only=[
        Path(sys.prefix),
        Path("/usr"),
        Path("/dev/null"),
        root,
        scripts,
    ],
    read_write=[state],
    cwd=scripts,
    env={
        "HOME": "/nonexistent-structsplat-sandbox-home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": __import__("os").environ["PATH"],
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(root),
    },
    timeout=120,
)
raise RuntimeError("middle unexpectedly exited before parent-death test")
""".lstrip(),
        encoding="utf-8",
    )
    controller_environment = dict(_NESTED_ENV)
    ambient_parent = os.environ.get(sandbox.SANDBOX_ATTESTATION_ENV)
    if ambient_parent is not None:
        # The controller is a direct descendant of the focused proof rather
        # than a new sandbox root, so preserve that already-attested lineage.
        controller_environment[sandbox.SANDBOX_ATTESTATION_ENV] = ambient_parent
    controller_process = subprocess.Popen(
        [
            sys.executable,
            "-B",
            os.fspath(controller),
            os.fspath(middle),
            os.fspath(leaf),
            os.fspath(state),
            os.fspath(_ROOT),
            os.fspath(scripts),
        ],
        cwd=scripts,
        env=controller_environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    middle_pid = -1
    leaf_pid = -1
    try:
        ready = _wait_until(
            lambda: (state / "middle.pid").is_file()
            and (state / "leaf.pid").is_file()
            and controller_process.poll() is None
        )
        if not ready:
            stdout, stderr = controller_process.communicate(timeout=5)
            pytest.fail(
                "nested cascade did not start: "
                f"returncode={controller_process.returncode}, "
                f"stdout={stdout!r}, stderr={stderr!r}"
            )
        middle_pid = int((state / "middle.pid").read_text(encoding="ascii"))
        leaf_pid = int((state / "leaf.pid").read_text(encoding="ascii"))
        os.kill(controller_process.pid, signal.SIGKILL)
        controller_process.communicate(timeout=10)
        assert _wait_until(lambda: _process_is_absent(middle_pid))
        assert _wait_until(lambda: _process_is_absent(leaf_pid))
    finally:
        if controller_process.poll() is None:
            controller_process.kill()
            controller_process.communicate(timeout=10)
        for pid in (middle_pid, leaf_pid):
            if pid > 0 and not _process_is_absent(pid):
                os.kill(pid, signal.SIGKILL)
