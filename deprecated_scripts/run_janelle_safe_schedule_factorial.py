#!/usr/bin/env python3
"""Run the source-bound 2x2 Janelle safe-schedule mechanism experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import fit_janelle_safe_commit_schedule as runner  # noqa: E402


DEFAULT_OUT = (
    REPOSITORY_ROOT
    / "runs/janelle_C0001_transactional_candidates_factorial_20260723"
)
ARMS: dict[str, dict[str, Any]] = {
    "control": {
        "label": "Kontrolle",
        "pareto_safe_checkpoints": False,
        "event_color_solve": False,
    },
    "pareto_checkpoint": {
        "label": "Pareto-Checkpoint",
        "pareto_safe_checkpoints": True,
        "event_color_solve": False,
    },
    "event_color_solve": {
        "label": "Event-Color-Solve",
        "pareto_safe_checkpoints": False,
        "event_color_solve": True,
    },
    "combined": {
        "label": "Kombiniert",
        "pareto_safe_checkpoints": True,
        "event_color_solve": True,
    },
}


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance() -> dict[str, Any]:
    sources = (
        Path(__file__).resolve(),
        SCRIPT_ROOT / "fit_janelle_safe_commit_schedule.py",
        SCRIPT_ROOT / "compare_janelle_safe_schedule_variants.py",
        REPOSITORY_ROOT / "src/structsplat/safe_schedule.py",
        REPOSITORY_ROOT / "src/structsplat/fit.py",
        REPOSITORY_ROOT / "src/structsplat/config.py",
    )
    status = _git("status", "--short")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "status": status,
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "sources": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sources
        ],
    }


def _arm_command(
    args: argparse.Namespace,
    arm_name: str,
    arm_out: Path,
) -> list[str]:
    arm = ARMS[arm_name]
    command = [
        sys.executable,
        str(SCRIPT_ROOT / "fit_janelle_safe_commit_schedule.py"),
        "--capture-root",
        str(args.capture_root.resolve()),
        "--realtime-root",
        str(args.realtime_root.resolve()),
        "--frame",
        args.frame,
        "--view-id",
        args.view_id,
        "--device",
        args.device,
        "--out",
        str(arm_out),
        "--preview-width",
        str(args.preview_width),
        "--pareto-checkpoint-every",
        str(args.pareto_checkpoint_every),
    ]
    if arm["pareto_safe_checkpoints"]:
        command.append("--pareto-safe-checkpoints")
    if arm["event_color_solve"]:
        command.append("--event-color-solve")
    if not args.with_archive:
        command.append("--no-archive")
    if args.smoke:
        command.extend([
            "--capacity", "5024",
            "--coverage-target", "5008",
            "--detail-target", "5016",
            "--bootstrap-steps", "2",
            "--coverage-steps", "2",
            "--detail-steps", "2",
            "--boundary-steps", "2",
            "--redistribution-steps", "2",
            "--polish-steps", "2",
            "--block-steps", "2",
            "--recovery-steps", "1",
            "--event-min-count", "1",
            "--coverage-birth-count", "8",
            "--detail-birth-count", "8",
            "--detail-split-count", "4",
            "--boundary-birth-count", "8",
            "--redistribution-count", "4",
        ])
    return command


def _complete_arm(path: Path, expected: dict[str, Any], args: argparse.Namespace) -> bool:
    required = (
        "summary.json",
        "run_config.json",
        "schedule_history.json",
        "index.html",
        "C0001_safe_commit_full.npz",
    )
    if not all((path / name).is_file() for name in required):
        return False
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    schedule = config["schedule"]
    checks = {
        "pareto_safe_checkpoints": bool(expected["pareto_safe_checkpoints"]),
        "event_color_solve": bool(expected["event_color_solve"]),
        "pareto_checkpoint_every": int(args.pareto_checkpoint_every),
    }
    for key, expected_value in checks.items():
        if schedule.get(key) != expected_value:
            raise RuntimeError(
                f"{path}: existing {key}={schedule.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    expected_source = f"{args.frame}/{args.view_id.upper()}"
    if summary.get("source") != expected_source:
        raise RuntimeError(
            f"{path}: existing source {summary.get('source')!r}, "
            f"expected {expected_source!r}"
        )
    return True


def _result_record(path: Path) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    history = json.loads((path / "schedule_history.json").read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "summary_sha256": _sha256(path / "summary.json"),
        "field_sha256": _sha256(path / "C0001_safe_commit_full.npz"),
        "metrics": summary["safe_metrics"],
        "attempted_steps": int(history["attempted_steps"]),
        "accepted_steps": int(history["accepted_steps"]),
        "converged": bool(history["converged"]),
        "total_seconds": float(config["timing"]["total_seconds"]),
        "peak_gpu_memory_bytes": int(config["timing"]["peak_gpu_memory_bytes"]),
    }


def run(args: argparse.Namespace) -> None:
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    requested = list(ARMS) if args.arm is None else list(dict.fromkeys(args.arm))
    commands = {
        name: _arm_command(args, name, out / name)
        for name in requested
    }
    manifest: dict[str, Any] = {
        "schema": "structsplat.janelle.safe_schedule_factorial.v1",
        "development_scope": "one Janelle image, one seed, one CUDA device",
        "frame": args.frame,
        "view_id": args.view_id.upper(),
        "device": args.device,
        "smoke": bool(args.smoke),
        "with_archive": bool(args.with_archive),
        "pareto_checkpoint_every": int(args.pareto_checkpoint_every),
        "arms": ARMS,
        "requested_arms": requested,
        "commands": commands,
        "repository": _source_provenance(),
        "results": {},
    }
    _atomic_json(out / "experiment_config.json", manifest)

    for name in requested:
        arm_out = out / name
        if _complete_arm(arm_out, ARMS[name], args):
            if not args.resume:
                raise RuntimeError(
                    f"{arm_out} is already complete; pass --resume to reuse it"
                )
            print(f"[factorial] reuse complete arm {name}: {arm_out}", flush=True)
        else:
            if arm_out.exists() and any(arm_out.iterdir()):
                raise RuntimeError(
                    f"{arm_out} is incomplete and non-empty; preserve or move it before rerun"
                )
            print(f"[factorial] run arm {name}: {' '.join(commands[name])}", flush=True)
            subprocess.run(commands[name], cwd=REPOSITORY_ROOT, check=True)
        manifest["results"][name] = _result_record(arm_out)
        _atomic_json(out / "experiment_config.json", manifest)

    complete = [
        name
        for name, expected in ARMS.items()
        if _complete_arm(out / name, expected, args)
    ]
    if len(complete) == len(ARMS):
        compare_command = [
            sys.executable,
            str(SCRIPT_ROOT / "compare_janelle_safe_schedule_variants.py"),
        ]
        for name, arm in ARMS.items():
            compare_command.extend([
                "--run",
                f"{arm['label']}={out / name}",
            ])
        compare_command.extend([
            "--out",
            str(out),
            "--require-factorial",
        ])
        print("[factorial] build audited comparison", flush=True)
        subprocess.run(compare_command, cwd=REPOSITORY_ROOT, check=True)
        manifest["comparison"] = {
            "path": str(out / "index.html"),
            "sha256": _sha256(out / "index.html"),
            "comparison_json_sha256": _sha256(out / "comparison.json"),
        }
        _atomic_json(out / "experiment_config.json", manifest)
    else:
        print(
            f"[factorial] {len(complete)}/4 arms complete; comparison deferred",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=runner.DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--realtime-root", type=Path, default=runner.DEFAULT_REALTIME_ROOT)
    parser.add_argument("--frame", default="frame_00008")
    parser.add_argument("--view-id", default="C0001")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preview-width", type=int, default=1200)
    parser.add_argument("--pareto-checkpoint-every", type=int, default=50)
    parser.add_argument("--with-archive", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", action="append", choices=tuple(ARMS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.preview_width <= 0 or args.pareto_checkpoint_every <= 0:
        raise ValueError("preview width and checkpoint cadence must be positive")
    run(args)


if __name__ == "__main__":
    main()
