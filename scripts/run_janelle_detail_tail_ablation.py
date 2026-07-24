#!/usr/bin/env python3
"""Run the matched-storage Janelle FIT-025 three-arm development experiment."""
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
    / "runs/janelle_C0001_detail_tail_ablation_20260724"
)
ARMS: dict[str, dict[str, Any]] = {
    "baseline_11000": {
        "label": "Fixed storage, 11k active",
        "extra_base_rows": 0,
        "detail_tail_rows": 0,
    },
    "generic_plus512": {
        "label": "Generic active budget +512",
        "extra_base_rows": 512,
        "detail_tail_rows": 0,
    },
    "adaptive_tail_plus512": {
        "label": "Adaptive detail tail +512",
        "extra_base_rows": 0,
        "detail_tail_rows": 512,
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
        REPOSITORY_ROOT / "src/structsplat/gaussians.py",
        REPOSITORY_ROOT / "src/structsplat/pool.py",
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


def _resolved_protocol(args: argparse.Namespace) -> dict[str, int]:
    if args.smoke:
        return {
            "capacity": 5_032,
            "base_active_limit": 5_016,
            "extra_rows": 8,
            "tail_batch": 4,
        }
    return {
        "capacity": int(args.capacity),
        "base_active_limit": int(args.base_active_limit),
        "extra_rows": int(args.extra_rows),
        "tail_batch": int(args.tail_batch),
    }


def _arm_command(
    args: argparse.Namespace,
    arm_name: str,
    arm_out: Path,
) -> list[str]:
    arm = ARMS[arm_name]
    protocol = _resolved_protocol(args)
    scale = protocol["extra_rows"] / 512
    extra_base_rows = round(int(arm["extra_base_rows"]) * scale)
    detail_tail_rows = round(int(arm["detail_tail_rows"]) * scale)
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
        "--storage-policy",
        "fixed_capacity",
        "--capacity",
        str(protocol["capacity"]),
        "--base-active-limit",
        str(protocol["base_active_limit"] + extra_base_rows),
        "--detail-tail-rows",
        str(detail_tail_rows),
        "--detail-tail-batch",
        str(protocol["tail_batch"]),
        "--detail-tail-min-gain-per-row",
        str(args.tail_min_gain_per_row),
        "--pareto-safe-checkpoints",
        "--pareto-checkpoint-every",
        str(args.pareto_checkpoint_every),
        "--no-archive",
    ]
    if args.smoke:
        command.extend([
            "--coverage-target", "5008",
            "--detail-target", "5012",
            "--bootstrap-steps", "2",
            "--coverage-steps", "2",
            "--detail-steps", "2",
            "--boundary-steps", "2",
            "--redistribution-steps", "2",
            "--polish-steps", "2",
            "--block-steps", "2",
            "--recovery-steps", "1",
            "--event-min-count", "1",
            "--coverage-birth-count", "4",
            "--detail-birth-count", "4",
            "--detail-split-count", "4",
            "--boundary-birth-count", "4",
            "--redistribution-count", "2",
        ])
    return command


def _expected_schedule(
    args: argparse.Namespace,
    arm_name: str,
) -> dict[str, Any]:
    command = _arm_command(args, arm_name, Path("unused"))

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    return {
        "storage_policy": "fixed_capacity",
        "capacity": int(value("--capacity")),
        "base_active_limit": int(value("--base-active-limit")),
        "detail_tail_max_rows": int(value("--detail-tail-rows")),
        "detail_tail_batch_rows": int(value("--detail-tail-batch")),
        "detail_tail_min_gain_per_row": float(
            value("--detail-tail-min-gain-per-row")
        ),
        "pareto_safe_checkpoints": True,
        "pareto_checkpoint_every": int(value("--pareto-checkpoint-every")),
    }


def _complete_arm(
    path: Path,
    args: argparse.Namespace,
    arm_name: str,
) -> bool:
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
    expected = _expected_schedule(args, arm_name)
    for key, expected_value in expected.items():
        actual = config["schedule"].get(key)
        if actual != expected_value:
            raise RuntimeError(
                f"{path}: existing {key}={actual!r}, expected {expected_value!r}"
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
    history = json.loads(
        (path / "schedule_history.json").read_text(encoding="utf-8")
    )
    tail_end = next(
        record
        for record in history["history"]
        if record["phase"] == "detail_tail"
        and record["event"] == "phase_end"
    )
    return {
        "path": str(path),
        "summary_sha256": _sha256(path / "summary.json"),
        "field_sha256": _sha256(path / "C0001_safe_commit_full.npz"),
        "metrics": summary["safe_metrics"],
        "attempted_steps": int(history["attempted_steps"]),
        "accepted_steps": int(history["accepted_steps"]),
        "converged": bool(history["converged"]),
        "schedule_seconds": float(history["seconds"]),
        "total_seconds": float(config["timing"]["total_seconds"]),
        "peak_gpu_memory_bytes": int(config["timing"]["peak_gpu_memory_bytes"]),
        "storage": history["storage"],
        "detail_tail": tail_end["metadata"],
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
        "schema": "structsplat.janelle.detail_tail_ablation.v1",
        "development_scope": "one Janelle image, one seed, one CUDA device",
        "frame": args.frame,
        "view_id": args.view_id.upper(),
        "device": args.device,
        "smoke": bool(args.smoke),
        "protocol": _resolved_protocol(args),
        "tail_min_gain_per_row": float(args.tail_min_gain_per_row),
        "arms": ARMS,
        "requested_arms": requested,
        "commands": commands,
        "repository": _source_provenance(),
        "results": {},
    }
    _atomic_json(out / "experiment_config.json", manifest)

    for name in requested:
        arm_out = out / name
        if _complete_arm(arm_out, args, name):
            if not args.resume:
                raise RuntimeError(
                    f"{arm_out} is already complete; pass --resume to reuse it"
                )
            print(f"[detail-tail] reuse complete arm {name}: {arm_out}", flush=True)
        else:
            if arm_out.exists() and any(arm_out.iterdir()):
                raise RuntimeError(
                    f"{arm_out} is incomplete and non-empty; preserve or move it before rerun"
                )
            print(
                f"[detail-tail] run arm {name}: {' '.join(commands[name])}",
                flush=True,
            )
            subprocess.run(commands[name], cwd=REPOSITORY_ROOT, check=True)
        manifest["results"][name] = _result_record(arm_out)
        _atomic_json(out / "experiment_config.json", manifest)

    complete = [
        name
        for name in ARMS
        if _complete_arm(out / name, args, name)
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
        compare_command.extend(["--out", str(out)])
        subprocess.run(compare_command, cwd=REPOSITORY_ROOT, check=True)
        manifest["comparison"] = {
            "path": str(out / "index.html"),
            "sha256": _sha256(out / "index.html"),
            "comparison_json_sha256": _sha256(out / "comparison.json"),
        }
        _atomic_json(out / "experiment_config.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-root", type=Path, default=runner.DEFAULT_CAPTURE_ROOT
    )
    parser.add_argument(
        "--realtime-root", type=Path, default=runner.DEFAULT_REALTIME_ROOT
    )
    parser.add_argument("--frame", default="frame_00008")
    parser.add_argument("--view-id", default="C0001")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preview-width", type=int, default=1200)
    parser.add_argument("--capacity", type=int, default=12_024)
    parser.add_argument("--base-active-limit", type=int, default=11_000)
    parser.add_argument("--extra-rows", type=int, default=512)
    parser.add_argument("--tail-batch", type=int, default=128)
    parser.add_argument("--tail-min-gain-per-row", type=float, default=0.0)
    parser.add_argument("--pareto-checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", action="append", choices=tuple(ARMS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    positive = (
        args.preview_width,
        args.capacity,
        args.base_active_limit,
        args.extra_rows,
        args.tail_batch,
        args.pareto_checkpoint_every,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sizes and checkpoint cadence must be positive")
    if args.base_active_limit + args.extra_rows > args.capacity:
        raise ValueError(
            "base active limit plus extra rows cannot exceed physical capacity"
        )
    if args.tail_min_gain_per_row < 0.0:
        raise ValueError("tail minimum gain per row must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
