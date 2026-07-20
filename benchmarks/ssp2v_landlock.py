"""Fail-closed Linux sandbox launcher for COMP-011 subprocesses.

The COMP-011 lifecycle runs captured Python source and several fresh workers.  A
parent-side path guard cannot observe (or prevent) filesystem opens performed by
those children.  This module installs two kernel restrictions before ``exec``:

* a Landlock allowlist for filesystem reads and writes, default-deny TCP
  bind/connect rules, and abstract-socket/signal scoping (ABI 6); and
* a seccomp filter that permits only ``AF_UNIX`` socket construction while
  denying outbound socket and cross-process syscall paths.

The restrictions are intentionally process-local and irreversible.  Callers
must launch them in a fresh subprocess.  The small public API returns a sealed
attestation through a dedicated pipe before replacing the launcher process with
the requested command.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import platform
import signal
import socket
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ATTESTATION_SCHEMA = "structsplat.comp011.landlock-attestation.v2"
SANDBOX_ATTESTATION_ENV = "STRUCTSPLAT_SANDBOX_ATTESTATION_SHA256"
RESTRICTION_MODE_FRESH = "fresh_root"
RESTRICTION_MODE_INHERITED = "inherited_nested"

_SOCKET_BASIS_DIRECT = "direct_bind_connect"
_SOCKET_BASIS_INHERITED = "inherited_non_unix_creation_denial"

_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REMOVE_DIR = 1 << 4
_FS_REMOVE_FILE = 1 << 5
_FS_MAKE_CHAR = 1 << 6
_FS_MAKE_DIR = 1 << 7
_FS_MAKE_REG = 1 << 8
_FS_MAKE_SOCK = 1 << 9
_FS_MAKE_FIFO = 1 << 10
_FS_MAKE_BLOCK = 1 << 11
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13
_FS_TRUNCATE = 1 << 14
_FS_IOCTL_DEV = 1 << 15

_FS_READ = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR
_FS_WRITE_SAFE = (
    _FS_WRITE_FILE
    | _FS_REMOVE_DIR
    | _FS_REMOVE_FILE
    | _FS_MAKE_DIR
    | _FS_MAKE_REG
    | _FS_MAKE_SOCK
    | _FS_MAKE_FIFO
    | _FS_MAKE_SYM
    | _FS_REFER
    | _FS_TRUNCATE
)
_FS_HANDLED = _FS_READ | _FS_WRITE_SAFE | _FS_MAKE_CHAR | _FS_MAKE_BLOCK | _FS_IOCTL_DEV

_NET_BIND_TCP = 1 << 0
_NET_CONNECT_TCP = 1 << 1
_NET_HANDLED = _NET_BIND_TCP | _NET_CONNECT_TCP

_SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0
_SCOPE_SIGNAL = 1 << 1
_SCOPE_HANDLED = _SCOPE_ABSTRACT_UNIX_SOCKET | _SCOPE_SIGNAL

_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_PR_GET_SECCOMP = 21
_PR_SET_PDEATHSIG = 1
_PR_GET_PDEATHSIG = 2
_SECCOMP_MODE_FILTER = 2
_SECCOMP_RET_KILL_PROCESS = 0x80000000
_SECCOMP_RET_ERRNO = 0x00050000
_SECCOMP_RET_ALLOW = 0x7FFF0000
_AUDIT_ARCH_X86_64 = 0xC000003E
_X86_64_NR_SOCKET = 41
_X86_64_NR_SOCKETPAIR = 53
_X32_SYSCALL_BIT = 0x40000000

# Syscalls that could communicate with, signal, inspect, or manipulate a
# process outside the sandbox after all inherited descriptors have been
# closed.  Socket creation itself is handled separately so private AF_UNIX
# socketpairs remain usable inside numerical runtimes.
_SECCOMP_DENIED_SYSCALLS = {
    "connect": 42,
    "sendto": 44,
    "sendmsg": 46,
    "ptrace": 101,
    "rt_sigqueueinfo": 129,
    "tkill": 200,
    "tgkill": 234,
    "rt_tgsigqueueinfo": 297,
    "perf_event_open": 298,
    "sendmmsg": 307,
    "process_vm_readv": 310,
    "process_vm_writev": 311,
    "kcmp": 312,
    "pidfd_send_signal": 424,
    "io_uring_setup": 425,
    "io_uring_enter": 426,
    "io_uring_register": 427,
    "pidfd_getfd": 438,
    "process_madvise": 440,
}

_BPF_LD_W_ABS = 0x20
_BPF_JMP_JEQ_K = 0x15
_BPF_JMP_JSET_K = 0x45
_BPF_RET_K = 0x06


class SandboxError(RuntimeError):
    """A kernel restriction or its precondition failed."""


class _RulesetAttrV1(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _RulesetAttrV4(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
    ]


class _RulesetAttrV6(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint16),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long
_LIBC.prctl.restype = ctypes.c_int


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SandboxError(f"sandbox attestation is not canonical JSON: {exc}") from exc


def _seal(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result.pop("attestation_sha256", None)
    result["attestation_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


def _digest_mapping(value: Mapping[str, str]) -> str:
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or "\x00" in key
            or "=" in key
            or not isinstance(item, str)
            or "\x00" in item
        ):
            raise SandboxError("sandbox environment must contain valid string keys/values")
        normalized[key] = item
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def verify_attestation(record: Mapping[str, Any]) -> None:
    """Verify the exact schema and canonical seal of a launcher attestation."""

    if set(record) != {
        "schema",
        "pid",
        "parent_pid",
        "restriction_mode",
        "parent_sandbox_attestation_sha256",
        "parent_death_signal",
        "parent_death_signal_verified",
        "parent_pid_race_checked",
        "landlock_abi",
        "filesystem_restricted",
        "tcp_bind_connect_restricted",
        "non_unix_socket_creation_restricted",
        "external_unix_ipc_restricted",
        "cross_process_restricted",
        "landlock_scope",
        "seccomp_denied_syscalls",
        "read_only",
        "read_write",
        "denied_read_probes",
        "socket_probe",
        "command",
        "command_sha256",
        "cwd",
        "environment_keys",
        "environment_sha256_before_attestation_marker",
        "launcher_sha256",
        "attestation_sha256",
    }:
        raise SandboxError("sandbox attestation key set drift")
    if record.get("schema") != ATTESTATION_SCHEMA:
        raise SandboxError("sandbox attestation schema drift")
    observed = record.get("attestation_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        raise SandboxError("sandbox attestation seal is absent")
    if _seal(record)["attestation_sha256"] != observed:
        raise SandboxError("sandbox attestation seal mismatch")
    if (
        record.get("filesystem_restricted") is not True
        or record.get("non_unix_socket_creation_restricted") is not True
        or record.get("tcp_bind_connect_restricted") is not True
        or record.get("external_unix_ipc_restricted") is not True
        or record.get("cross_process_restricted") is not True
    ):
        raise SandboxError("sandbox attestation does not prove all required restrictions")
    if (
        not isinstance(record.get("landlock_abi"), int)
        or isinstance(record.get("landlock_abi"), bool)
        or int(record["landlock_abi"]) < 6
        or record.get("landlock_scope") != ["abstract_unix_socket", "signal"]
        or record.get("seccomp_denied_syscalls") != sorted(_SECCOMP_DENIED_SYSCALLS)
    ):
        raise SandboxError("sandbox IPC restriction inventory drift")
    restriction_mode = record.get("restriction_mode")
    parent_attestation = record.get("parent_sandbox_attestation_sha256")
    parent_digest_is_valid = (
        isinstance(parent_attestation, str)
        and len(parent_attestation) == 64
        and all(character in "0123456789abcdef" for character in parent_attestation)
    )
    if (
        restriction_mode not in {RESTRICTION_MODE_FRESH, RESTRICTION_MODE_INHERITED}
        or (
            restriction_mode == RESTRICTION_MODE_FRESH
            and parent_attestation is not None
        )
        or (
            restriction_mode == RESTRICTION_MODE_INHERITED
            and not parent_digest_is_valid
        )
        or record.get("parent_death_signal") != int(signal.SIGKILL)
        or record.get("parent_death_signal_verified") is not True
        or record.get("parent_pid_race_checked") is not True
    ):
        raise SandboxError("sandbox ancestry or parent-death binding is malformed")
    command = record.get("command")
    if (
        not isinstance(record.get("pid"), int)
        or isinstance(record.get("pid"), bool)
        or int(record["pid"]) <= 0
        or not isinstance(record.get("parent_pid"), int)
        or isinstance(record.get("parent_pid"), bool)
        or int(record["parent_pid"]) <= 0
        or not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
        or record.get("command_sha256") != hashlib.sha256(_canonical_json(command)).hexdigest()
        or not isinstance(record.get("cwd"), str)
        or not Path(str(record["cwd"])).is_absolute()
        or not isinstance(record.get("environment_keys"), list)
        or any(not isinstance(item, str) or not item for item in record["environment_keys"])
        or record.get("environment_keys") != sorted(set(record["environment_keys"]))
    ):
        raise SandboxError("sandbox execution binding is malformed")
    for field in ("read_only", "read_write"):
        paths = record.get(field)
        if (
            not isinstance(paths, list)
            or any(not isinstance(path, str) or not Path(path).is_absolute() for path in paths)
            or paths != sorted(set(paths))
        ):
            raise SandboxError(f"sandbox {field} inventory is malformed")
    denied_read_probes = record.get("denied_read_probes")
    if (
        not isinstance(denied_read_probes, list)
        or any(
            not isinstance(probe, Mapping)
            or set(probe) != {"path", "errno"}
            or not isinstance(probe.get("path"), str)
            or not Path(str(probe["path"])).is_absolute()
            or probe.get("errno") not in (errno.EACCES, errno.EPERM)
            for probe in denied_read_probes
        )
        or denied_read_probes != sorted(denied_read_probes, key=lambda probe: str(probe["path"]))
        or len({str(probe["path"]) for probe in denied_read_probes}) != len(denied_read_probes)
    ):
        raise SandboxError("sandbox denied-read probe inventory is malformed")
    for name in (
        "environment_sha256_before_attestation_marker",
        "launcher_sha256",
    ):
        digest = record.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SandboxError(f"sandbox {name} is malformed")
    socket_probe = record.get("socket_probe")
    if not isinstance(socket_probe, Mapping) or set(socket_probe) != {
        "basis",
        "inherited_AF_INET_create",
        "inherited_AF_INET6_create",
        "TCP_bind",
        "TCP_connect",
        "AF_INET",
        "AF_INET6",
        "AF_UNIX_allowed",
    }:
        raise SandboxError("sandbox socket probes do not prove the frozen restrictions")
    denial_errnos = (errno.EACCES, errno.EPERM)
    common_socket_evidence = (
        socket_probe.get("AF_UNIX_allowed") == 1
        and socket_probe.get("AF_INET") in denial_errnos
        and socket_probe.get("AF_INET6") in denial_errnos
    )
    direct_socket_evidence = (
        restriction_mode == RESTRICTION_MODE_FRESH
        and socket_probe.get("basis") == _SOCKET_BASIS_DIRECT
        and socket_probe.get("inherited_AF_INET_create") is None
        and socket_probe.get("inherited_AF_INET6_create") is None
        and socket_probe.get("TCP_bind") in denial_errnos
        and socket_probe.get("TCP_connect") in denial_errnos
    )
    inherited_socket_evidence = (
        restriction_mode == RESTRICTION_MODE_INHERITED
        and socket_probe.get("basis") == _SOCKET_BASIS_INHERITED
        and socket_probe.get("inherited_AF_INET_create") in denial_errnos
        and socket_probe.get("inherited_AF_INET6_create") in denial_errnos
        and socket_probe.get("TCP_bind") is None
        and socket_probe.get("TCP_connect") is None
    )
    if not common_socket_evidence or not (
        direct_socket_evidence or inherited_socket_evidence
    ):
        raise SandboxError("sandbox socket probes do not prove the frozen restrictions")


def landlock_abi() -> int:
    """Return the kernel Landlock ABI, or raise instead of silently weakening."""

    ctypes.set_errno(0)
    result = _LIBC.syscall(
        ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    if result < 1:
        error = ctypes.get_errno()
        raise SandboxError(f"Landlock ABI query failed: errno={error} ({os.strerror(error)})")
    return int(result)


def _resolved_allowlist(paths: Sequence[Path], name: str) -> tuple[Path, ...]:
    resolved: set[Path] = set()
    for index, raw in enumerate(paths):
        path = Path(raw)
        if not path.is_absolute():
            raise SandboxError(f"{name}[{index}] must be absolute")
        try:
            canonical = path.resolve(strict=True)
        except OSError as exc:
            raise SandboxError(f"{name}[{index}] is not an existing path: {path}") from exc
        resolved.add(canonical)
    return tuple(sorted(resolved, key=os.fspath))


def _lexical_probe_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Validate denied probes without touching their forbidden metadata."""

    probes: set[Path] = set()
    for index, raw in enumerate(paths):
        path = Path(raw)
        if not path.is_absolute():
            raise SandboxError(f"denied_read_probes[{index}] must be absolute")
        if any(part in ("", ".", "..") for part in path.parts[1:]):
            raise SandboxError(f"denied_read_probes[{index}] is not lexical-normal")
        probes.add(path)
    return tuple(sorted(probes, key=os.fspath))


def _syscall_error(result: int, operation: str) -> None:
    if result < 0:
        error = ctypes.get_errno()
        raise SandboxError(f"{operation} failed: errno={error} ({os.strerror(error)})")


def _add_path_rule(ruleset_fd: int, path: Path, *, writable: bool) -> None:
    mode = path.stat().st_mode
    is_directory = stat.S_ISDIR(mode)
    is_regular = stat.S_ISREG(mode)
    is_device = stat.S_ISCHR(mode) or stat.S_ISBLK(mode)
    if not is_directory and not is_regular and not is_device:
        raise SandboxError(f"allowlist path must be a directory, regular file, or device: {path}")
    flags = os.O_PATH | os.O_CLOEXEC
    descriptor = os.open(path, flags)
    try:
        if is_directory:
            access = _FS_READ | (_FS_WRITE_SAFE if writable else 0)
        elif is_regular:
            access = _FS_EXECUTE | _FS_READ_FILE
            if writable:
                access |= _FS_WRITE_FILE | _FS_TRUNCATE
        else:
            # CUDA workers need exact O_RDWR access to sealed device nodes such
            # as /dev/nvidia0.  Grant no create/remove/refer rights here.
            access = _FS_READ_FILE | _FS_IOCTL_DEV | (_FS_WRITE_FILE if writable else 0)
        attribute = _PathBeneathAttr(access, descriptor)
        ctypes.set_errno(0)
        result = _LIBC.syscall(
            ctypes.c_long(_SYS_LANDLOCK_ADD_RULE),
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(attribute),
            ctypes.c_uint(0),
        )
        _syscall_error(result, f"Landlock add-rule for {path}")
    finally:
        os.close(descriptor)


def _set_no_new_privileges() -> None:
    ctypes.set_errno(0)
    result = _LIBC.prctl(
        ctypes.c_int(_PR_SET_NO_NEW_PRIVS),
        ctypes.c_ulong(1),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    _syscall_error(result, "prctl(PR_SET_NO_NEW_PRIVS)")


def _arm_parent_death_signal(expected_parent_pid: int) -> None:
    """Kill this launcher if its exact spawning parent disappears.

    Passing the parent's PID through the launch vector closes the usual race
    where a child first observes PID 1 after its parent has already exited.
    The second check closes the race between the initial observation and
    ``PR_SET_PDEATHSIG``.
    """

    if (
        isinstance(expected_parent_pid, bool)
        or not isinstance(expected_parent_pid, int)
        or expected_parent_pid <= 1
        or os.getppid() != expected_parent_pid
    ):
        raise SandboxError("sandbox launcher parent changed before PDEATHSIG arm")
    ctypes.set_errno(0)
    result = _LIBC.prctl(
        ctypes.c_int(_PR_SET_PDEATHSIG),
        ctypes.c_ulong(int(signal.SIGKILL)),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    _syscall_error(result, "prctl(PR_SET_PDEATHSIG)")
    observed_signal = ctypes.c_int(0)
    ctypes.set_errno(0)
    result = _LIBC.prctl(
        ctypes.c_int(_PR_GET_PDEATHSIG),
        ctypes.byref(observed_signal),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    _syscall_error(result, "prctl(PR_GET_PDEATHSIG)")
    if observed_signal.value != int(signal.SIGKILL):
        raise SandboxError("PR_GET_PDEATHSIG did not verify SIGKILL")
    if os.getppid() != expected_parent_pid:
        # The kernel should already have delivered SIGKILL if the race landed
        # after the prctl.  Explicitly terminate if userspace observes the
        # changed parent first; continuing without the bound parent is never
        # acceptable.
        os.kill(os.getpid(), signal.SIGKILL)
        os._exit(128 + int(signal.SIGKILL))  # pragma: no cover


def _current_seccomp_mode() -> int:
    ctypes.set_errno(0)
    result = _LIBC.prctl(
        ctypes.c_int(_PR_GET_SECCOMP),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    _syscall_error(result, "prctl(PR_GET_SECCOMP)")
    return int(result)


def _install_seccomp_filter() -> None:
    if platform.machine() != "x86_64":
        raise SandboxError("COMP-011 seccomp filter is frozen for x86_64 only")

    # struct seccomp_data offsets: nr=0, arch=4, args[0]=16.  Reject the x32
    # syscall namespace before comparing native syscall numbers.  Each exact
    # deny is encoded as a two-instruction pair; all jumps therefore stay local
    # and do not depend on the size of the inventory.
    raw_instructions = [
        _SockFilter(_BPF_LD_W_ABS, 0, 0, 4),
        _SockFilter(_BPF_JMP_JEQ_K, 1, 0, _AUDIT_ARCH_X86_64),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_KILL_PROCESS),
        _SockFilter(_BPF_LD_W_ABS, 0, 0, 0),
        _SockFilter(_BPF_JMP_JSET_K, 0, 1, _X32_SYSCALL_BIT),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | errno.EACCES),
    ]
    for syscall_number in sorted(_SECCOMP_DENIED_SYSCALLS.values()):
        raw_instructions.extend(
            (
                _SockFilter(_BPF_JMP_JEQ_K, 0, 1, syscall_number),
                _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | errno.EACCES),
            )
        )

    # Permit local AF_UNIX construction, including private socketpairs used by
    # some numerical libraries, while rejecting every other socket domain.
    raw_instructions.extend(
        (
            _SockFilter(_BPF_JMP_JEQ_K, 0, 4, _X86_64_NR_SOCKET),
            _SockFilter(_BPF_LD_W_ABS, 0, 0, 16),
            _SockFilter(_BPF_JMP_JEQ_K, 1, 0, socket.AF_UNIX),
            _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | errno.EACCES),
            _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW),
            _SockFilter(_BPF_JMP_JEQ_K, 0, 4, _X86_64_NR_SOCKETPAIR),
            _SockFilter(_BPF_LD_W_ABS, 0, 0, 16),
            _SockFilter(_BPF_JMP_JEQ_K, 1, 0, socket.AF_UNIX),
            _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | errno.EACCES),
            _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW),
            _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW),
        )
    )
    instructions = (_SockFilter * len(raw_instructions))(*raw_instructions)
    program = _SockFprog(len(instructions), instructions)
    ctypes.set_errno(0)
    result = _LIBC.prctl(
        ctypes.c_int(_PR_SET_SECCOMP),
        ctypes.c_ulong(_SECCOMP_MODE_FILTER),
        ctypes.byref(program),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    _syscall_error(result, "prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER)")


def restrict_current_process(
    *,
    read_only: Sequence[Path],
    read_write: Sequence[Path],
    denied_read_probes: Sequence[Path] = (),
    command_binding: Sequence[str],
    cwd_binding: Path,
    environment_binding: Mapping[str, str],
    launcher_sha256: str,
    restriction_mode: str,
    parent_sandbox_attestation_sha256: str | None,
    expected_parent_pid: int,
) -> dict[str, Any]:
    """Install the frozen restrictions and return a sealed, probed attestation.

    ``read_only`` and ``read_write`` paths must exist and be absolute.  A
    read-write rule grants ordinary file/directory mutation but deliberately
    does not grant character- or block-device creation.
    """

    command = list(command_binding)
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise SandboxError("command binding must contain nonempty strings")
    if len(launcher_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in launcher_sha256
    ):
        raise SandboxError("launcher SHA-256 binding is malformed")
    cwd = Path(cwd_binding)
    if not cwd.is_absolute():
        raise SandboxError("cwd binding must be absolute")
    parent_digest_is_valid = (
        isinstance(parent_sandbox_attestation_sha256, str)
        and len(parent_sandbox_attestation_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in parent_sandbox_attestation_sha256
        )
    )
    if (
        restriction_mode not in {RESTRICTION_MODE_FRESH, RESTRICTION_MODE_INHERITED}
        or (
            restriction_mode == RESTRICTION_MODE_FRESH
            and parent_sandbox_attestation_sha256 is not None
        )
        or (
            restriction_mode == RESTRICTION_MODE_INHERITED
            and not parent_digest_is_valid
        )
    ):
        raise SandboxError("sandbox restriction mode and parent digest disagree")

    # Arm this before ABI queries, path resolution, probes, or policy setup.
    # PDEATHSIG is preserved across exec but cleared by fork; every recursive
    # sandbox launcher therefore re-arms and verifies its own parent binding.
    _arm_parent_death_signal(expected_parent_pid)
    if (
        restriction_mode == RESTRICTION_MODE_INHERITED
        and _current_seccomp_mode() != _SECCOMP_MODE_FILTER
    ):
        raise SandboxError("nested launcher did not inherit seccomp filter mode")
    abi = landlock_abi()
    environment_sha256 = _digest_mapping(environment_binding)
    readonly = _resolved_allowlist(read_only, "read_only")
    readwrite = _resolved_allowlist(read_write, "read_write")
    # Resolving or statting a forbidden target before restrict-self would itself
    # violate the lifecycle this probe is meant to prove.
    probes = _lexical_probe_paths(denied_read_probes)
    if set(readonly) & set(readwrite):
        raise SandboxError("a path cannot be both read-only and read-write")

    # A fresh root can create two inert TCP sockets before restriction and use
    # them to directly prove Landlock's bind/connect mediation.  A nested
    # launcher first verifies PR_GET_SECCOMP==FILTER and therefore cannot create
    # those sockets at all.  In that mode, record the actual AF_INET/AF_INET6
    # creation denials instead of claiming bind/connect calls that never ran.
    socket_probes: dict[str, Any] = {
        "basis": (
            _SOCKET_BASIS_DIRECT
            if restriction_mode == RESTRICTION_MODE_FRESH
            else _SOCKET_BASIS_INHERITED
        ),
        "inherited_AF_INET_create": None,
        "inherited_AF_INET6_create": None,
        "TCP_bind": None,
        "TCP_connect": None,
    }
    tcp_bind_probe: socket.socket | None = None
    tcp_connect_probe: socket.socket | None = None
    if restriction_mode == RESTRICTION_MODE_FRESH:
        try:
            tcp_bind_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_connect_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except BaseException:
            if tcp_bind_probe is not None:
                tcp_bind_probe.close()
            raise
    else:
        for family, name in (
            (socket.AF_INET, "inherited_AF_INET_create"),
            (socket.AF_INET6, "inherited_AF_INET6_create"),
        ):
            try:
                value = socket.socket(family, socket.SOCK_STREAM)
            except PermissionError as exc:
                if exc.errno not in (errno.EACCES, errno.EPERM):
                    raise SandboxError(
                        f"{name} probe failed with an unexpected errno"
                    ) from exc
                socket_probes[name] = int(exc.errno)
            else:
                value.close()
                raise SandboxError(
                    "nested launcher lacks the claimed inherited socket restriction"
                )

    try:
        if abi >= 6:
            ruleset_attribute: ctypes.Structure = _RulesetAttrV6(
                _FS_HANDLED, _NET_HANDLED, _SCOPE_HANDLED
            )
        else:
            raise SandboxError(
                f"Landlock ABI {abi} cannot enforce the frozen filesystem, "
                "network, abstract-socket, and signal restrictions"
            )
        ctypes.set_errno(0)
        ruleset_fd = _LIBC.syscall(
            ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
            ctypes.byref(ruleset_attribute),
            ctypes.c_size_t(ctypes.sizeof(ruleset_attribute)),
            ctypes.c_uint(0),
        )
        _syscall_error(ruleset_fd, "Landlock create-ruleset")
        try:
            for path in readonly:
                _add_path_rule(int(ruleset_fd), path, writable=False)
            for path in readwrite:
                _add_path_rule(int(ruleset_fd), path, writable=True)
            _set_no_new_privileges()
            ctypes.set_errno(0)
            result = _LIBC.syscall(
                ctypes.c_long(_SYS_LANDLOCK_RESTRICT_SELF),
                ctypes.c_int(ruleset_fd),
                ctypes.c_uint(0),
            )
            _syscall_error(result, "Landlock restrict-self")
        finally:
            os.close(int(ruleset_fd))

        if tcp_bind_probe is not None and tcp_connect_probe is not None:
            try:
                tcp_bind_probe.bind(("127.0.0.1", 0))
            except PermissionError as exc:
                if exc.errno not in (errno.EACCES, errno.EPERM):
                    raise SandboxError("TCP bind probe failed unexpectedly") from exc
                socket_probes["TCP_bind"] = int(exc.errno)
            else:
                raise SandboxError("TCP bind unexpectedly succeeded")
            try:
                tcp_connect_probe.connect(("127.0.0.1", 9))
            except PermissionError as exc:
                if exc.errno not in (errno.EACCES, errno.EPERM):
                    raise SandboxError("TCP connect probe failed unexpectedly") from exc
                socket_probes["TCP_connect"] = int(exc.errno)
            else:
                raise SandboxError("TCP connect unexpectedly succeeded")
    finally:
        if tcp_bind_probe is not None:
            tcp_bind_probe.close()
        if tcp_connect_probe is not None:
            tcp_connect_probe.close()

    _install_seccomp_filter()

    probe_records = []
    for path in probes:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        except PermissionError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM):
                raise SandboxError(f"denied-read probe failed unexpectedly: {path}") from exc
            probe_records.append({"path": os.fspath(path), "errno": int(exc.errno)})
        else:
            os.close(descriptor)
            raise SandboxError(f"forbidden read probe unexpectedly succeeded: {path}")

    for family, name in ((socket.AF_INET, "AF_INET"), (socket.AF_INET6, "AF_INET6")):
        try:
            value = socket.socket(family, socket.SOCK_STREAM)
        except PermissionError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM):
                raise SandboxError(f"{name} socket probe failed unexpectedly") from exc
            socket_probes[name] = int(exc.errno)
        else:
            value.close()
            raise SandboxError(f"{name} socket creation unexpectedly succeeded")
    unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    unix_socket.close()
    socket_probes["AF_UNIX_allowed"] = 1

    return _seal(
        {
            "schema": ATTESTATION_SCHEMA,
            "pid": os.getpid(),
            "parent_pid": expected_parent_pid,
            "restriction_mode": restriction_mode,
            "parent_sandbox_attestation_sha256": parent_sandbox_attestation_sha256,
            "parent_death_signal": int(signal.SIGKILL),
            "parent_death_signal_verified": True,
            "parent_pid_race_checked": True,
            "landlock_abi": abi,
            "filesystem_restricted": True,
            "tcp_bind_connect_restricted": True,
            "non_unix_socket_creation_restricted": True,
            "external_unix_ipc_restricted": True,
            "cross_process_restricted": True,
            "landlock_scope": ["abstract_unix_socket", "signal"],
            "seccomp_denied_syscalls": sorted(_SECCOMP_DENIED_SYSCALLS),
            "read_only": [os.fspath(path) for path in readonly],
            "read_write": [os.fspath(path) for path in readwrite],
            "denied_read_probes": probe_records,
            "socket_probe": socket_probes,
            "command": command,
            "command_sha256": hashlib.sha256(_canonical_json(command)).hexdigest(),
            "cwd": os.fspath(cwd),
            "environment_keys": sorted(environment_binding),
            "environment_sha256_before_attestation_marker": environment_sha256,
            "launcher_sha256": launcher_sha256,
        }
    )


def _launcher_arguments(
    command: Sequence[str],
    *,
    read_only: Sequence[Path],
    read_write: Sequence[Path],
    denied_read_probes: Sequence[Path],
    attestation_fd: int,
    launcher_path: Path,
    restriction_mode: str,
    parent_sandbox_attestation_sha256: str | None,
    expected_parent_pid: int,
) -> list[str]:
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise SandboxError("sandbox command must contain nonempty strings")
    arguments = [
        sys.executable,
        "-I",
        "-B",
        os.fspath(launcher_path.resolve(strict=True)),
        "_exec",
        "--attestation-fd",
        str(attestation_fd),
        "--restriction-mode",
        restriction_mode,
        "--expected-parent-pid",
        str(expected_parent_pid),
    ]
    if parent_sandbox_attestation_sha256 is not None:
        arguments.extend(
            (
                "--parent-sandbox-attestation-sha256",
                parent_sandbox_attestation_sha256,
            )
        )
    for path in read_only:
        arguments.extend(("--read-only", os.fspath(path)))
    for path in read_write:
        arguments.extend(("--read-write", os.fspath(path)))
    for path in denied_read_probes:
        arguments.extend(("--probe-denied-read", os.fspath(path)))
    arguments.append("--")
    arguments.extend(command)
    return arguments


def _prepare_expected_allowlist(paths: Sequence[Path], name: str) -> tuple[tuple[str, str], ...]:
    """Resolve a parent request while preserving child-relative ``/proc/self``."""

    prepared: set[tuple[str, str]] = set()
    proc_self = Path("/proc/self")
    for index, raw in enumerate(paths):
        path = Path(raw)
        if not path.is_absolute():
            raise SandboxError(f"{name}[{index}] must be absolute")
        if path.is_relative_to(proc_self):
            # Validate existence in the parent, but bind the attestation to the
            # launcher PID that /proc/self denotes inside the child.
            try:
                path.resolve(strict=True)
            except OSError as exc:
                raise SandboxError(f"{name}[{index}] is not an existing path: {path}") from exc
            relative = path.relative_to(proc_self)
            prepared.add(("proc-self", relative.as_posix()))
            continue
        try:
            canonical = path.resolve(strict=True)
        except OSError as exc:
            raise SandboxError(f"{name}[{index}] is not an existing path: {path}") from exc
        prepared.add(("path", os.fspath(canonical)))
    return tuple(sorted(prepared))


def _materialize_expected_allowlist(
    prepared: Sequence[tuple[str, str]], child_pid: int
) -> list[str]:
    values: set[str] = set()
    for kind, value in prepared:
        if kind == "proc-self":
            path = Path("/proc") / str(child_pid)
            if value != ".":
                path /= value
            values.add(os.fspath(path))
        elif kind == "path":
            values.add(value)
        else:  # pragma: no cover - private invariant
            raise AssertionError(f"unknown prepared allowlist kind: {kind}")
    return sorted(values)


def run_sandboxed(
    command: Sequence[str],
    *,
    read_only: Sequence[Path],
    read_write: Sequence[Path],
    denied_read_probes: Sequence[Path] = (),
    cwd: Path | None = None,
    env: Mapping[str, str],
    timeout: float | None = None,
    launcher_path: Path | None = None,
) -> tuple[subprocess.CompletedProcess[bytes], dict[str, Any]]:
    """Run ``command`` behind the kernel restrictions and return its attestation."""

    environment = dict(env)
    # This marker is launcher-owned lineage state, not caller-controlled worker
    # configuration.  A nested caller reads its own marker out-of-band; the
    # child receives a replacement containing its new attestation digest.
    if SANDBOX_ATTESTATION_ENV in environment:
        raise SandboxError(
            "requested child environment contains reserved attestation marker"
        )
    parent_attestation = os.environ.get(SANDBOX_ATTESTATION_ENV)
    if parent_attestation is None:
        restriction_mode = RESTRICTION_MODE_FRESH
    elif (
        len(parent_attestation) == 64
        and all(character in "0123456789abcdef" for character in parent_attestation)
    ):
        restriction_mode = RESTRICTION_MODE_INHERITED
    else:
        raise SandboxError("caller sandbox-attestation marker is malformed")
    expected_environment_sha256 = _digest_mapping(environment)
    resolved_cwd = (cwd or Path.cwd()).resolve(strict=True)
    resolved_launcher = (launcher_path or Path(__file__)).resolve(strict=True)
    expected_launcher_sha256 = hashlib.sha256(resolved_launcher.read_bytes()).hexdigest()
    prepared_readonly = _prepare_expected_allowlist(read_only, "read_only")
    prepared_readwrite = _prepare_expected_allowlist(read_write, "read_write")
    expected_denied_paths = [os.fspath(path) for path in _lexical_probe_paths(denied_read_probes)]
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    os.set_inheritable(write_fd, True)
    process: subprocess.Popen[bytes] | None = None
    try:
        arguments = _launcher_arguments(
            command,
            read_only=read_only,
            read_write=read_write,
            denied_read_probes=denied_read_probes,
            attestation_fd=write_fd,
            launcher_path=resolved_launcher,
            restriction_mode=restriction_mode,
            parent_sandbox_attestation_sha256=parent_attestation,
            expected_parent_pid=os.getpid(),
        )
        devnull_fd = os.open(
            os.devnull,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            if not stat.S_ISCHR(os.fstat(devnull_fd).st_mode):
                raise SandboxError("os.devnull is not a character device")
            process = subprocess.Popen(
                arguments,
                cwd=resolved_cwd,
                env=environment,
                stdin=devnull_fd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(write_fd,),
            )
        finally:
            os.close(devnull_fd)
        expected_readonly = _materialize_expected_allowlist(prepared_readonly, process.pid)
        expected_readwrite = _materialize_expected_allowlist(prepared_readwrite, process.pid)
        os.close(write_fd)
        write_fd = -1
        stdout, stderr = process.communicate(timeout=timeout)
        with os.fdopen(read_fd, "rb", closefd=True) as handle:
            attestation_payload = handle.read()
        read_fd = -1
    except BaseException as exc:
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=30)
            except ProcessLookupError:
                # A concurrent natural exit is equivalent to a successful
                # termination; wait() below has already been won elsewhere.
                process.wait(timeout=30)
            except BaseException as cleanup_exc:
                exc.add_note(
                    "sandbox child cleanup failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
        raise
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        if read_fd >= 0:
            os.close(read_fd)
    if process is None:  # pragma: no cover - successful path invariant
        raise AssertionError("sandbox process was not created")
    completed = subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
    try:
        attestation = json.loads(attestation_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SandboxError(
            "sandbox launcher did not emit one valid attestation: "
            f"returncode={completed.returncode}, stderr={stderr!r}"
        ) from exc
    if _canonical_json(attestation) != attestation_payload:
        raise SandboxError("sandbox launcher attestation is not canonical")
    verify_attestation(attestation)
    if (
        attestation.get("command") != list(command)
        or attestation.get("pid") != process.pid
        or attestation.get("cwd") != os.fspath(resolved_cwd)
        or attestation.get("environment_keys") != sorted(environment)
        or attestation.get("environment_sha256_before_attestation_marker")
        != expected_environment_sha256
        or attestation.get("launcher_sha256") != expected_launcher_sha256
        or attestation.get("parent_pid") != os.getpid()
        or attestation.get("restriction_mode") != restriction_mode
        or attestation.get("parent_sandbox_attestation_sha256")
        != parent_attestation
        or attestation.get("read_only") != expected_readonly
        or attestation.get("read_write") != expected_readwrite
        or [probe["path"] for probe in attestation.get("denied_read_probes", [])]
        != expected_denied_paths
    ):
        raise SandboxError("sandbox execution attestation differs from the exact request")
    return completed, dict(attestation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    child = subparsers.add_parser("_exec")
    child.add_argument("--attestation-fd", type=int, required=True)
    child.add_argument(
        "--restriction-mode",
        choices=(RESTRICTION_MODE_FRESH, RESTRICTION_MODE_INHERITED),
        required=True,
    )
    child.add_argument("--parent-sandbox-attestation-sha256")
    child.add_argument("--expected-parent-pid", type=int, required=True)
    child.add_argument("--read-only", action="append", type=Path, default=[])
    child.add_argument("--read-write", action="append", type=Path, default=[])
    child.add_argument("--probe-denied-read", action="append", type=Path, default=[])
    child.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SandboxError("sandbox launcher requires a command after --")
    cwd = Path.cwd().resolve(strict=True)
    environment = os.environ.copy()
    launcher_sha256 = hashlib.sha256(Path(__file__).resolve(strict=True).read_bytes()).hexdigest()
    attestation = restrict_current_process(
        read_only=args.read_only,
        read_write=args.read_write,
        denied_read_probes=args.probe_denied_read,
        command_binding=command,
        cwd_binding=cwd,
        environment_binding=environment,
        launcher_sha256=launcher_sha256,
        restriction_mode=args.restriction_mode,
        parent_sandbox_attestation_sha256=args.parent_sandbox_attestation_sha256,
        expected_parent_pid=args.expected_parent_pid,
    )
    payload = _canonical_json(attestation)
    written = os.write(args.attestation_fd, payload)
    if written != len(payload):
        raise SandboxError("sandbox attestation pipe write was short")
    os.close(args.attestation_fd)
    environment[SANDBOX_ATTESTATION_ENV] = str(attestation["attestation_sha256"])
    os.execvpe(command[0], command, environment)
    raise AssertionError("os.execvpe returned")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTESTATION_SCHEMA",
    "SANDBOX_ATTESTATION_ENV",
    "RESTRICTION_MODE_FRESH",
    "RESTRICTION_MODE_INHERITED",
    "SandboxError",
    "landlock_abi",
    "restrict_current_process",
    "run_sandboxed",
    "verify_attestation",
]
