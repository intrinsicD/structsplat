#!/usr/bin/env python3
"""Run a command and record process-level resource telemetry.

The benchmark harnesses already emit per-cell quality and timing rows. This wrapper adds
run-level resource provenance: wall time, child max RSS from ``resource.getrusage``, and sampled
GPU memory from ``nvidia-smi`` when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import signal
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sample_gpu(stop: threading.Event, rows: list[dict[str, Any]], interval: float) -> None:
    if shutil.which("nvidia-smi") is None:
        return
    query = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    while not stop.is_set():
        try:
            completed = subprocess.run(
                query,
                text=True,
                capture_output=True,
                check=False,
                timeout=max(1.0, interval),
            )
            if completed.returncode == 0:
                monotonic = time.monotonic()
                for line in completed.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 5:
                        continue
                    timestamp, index, name, used, total = parts
                    try:
                        rows.append(
                            {
                                "sample_monotonic_s": monotonic,
                                "timestamp": timestamp,
                                "gpu_index": int(index),
                                "gpu_name": name,
                                "memory_used_mib": int(used),
                                "memory_total_mib": int(total),
                            }
                        )
                    except ValueError:
                        continue
        except Exception:
            pass
        stop.wait(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command after --")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    stdout_path = outdir / f"{args.name}.stdout.log"
    stderr_path = outdir / f"{args.name}.stderr.log"
    gpu_path = outdir / f"{args.name}.gpu_memory.csv"
    summary_path = outdir / f"{args.name}.resources.json"

    gpu_rows: list[dict[str, Any]] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_gpu,
        args=(stop, gpu_rows, max(float(args.gpu_sample_interval), 0.1)),
        daemon=True,
    )

    start_wall = time.perf_counter()
    start_iso = datetime.now(timezone.utc).isoformat()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    interrupted = False
    sampler.start()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            returncode = proc.wait()
        except KeyboardInterrupt:
            interrupted = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = proc.wait()
    stop.set()
    sampler.join(timeout=2.0)
    end_iso = datetime.now(timezone.utc).isoformat()
    elapsed = time.perf_counter() - start_wall
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)

    if gpu_rows:
        with gpu_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(gpu_rows[0]))
            writer.writeheader()
            writer.writerows(gpu_rows)

    by_gpu: dict[int, dict[str, Any]] = {}
    for row in gpu_rows:
        idx = int(row["gpu_index"])
        current = by_gpu.setdefault(
            idx,
            {
                "gpu_index": idx,
                "gpu_name": row["gpu_name"],
                "memory_total_mib": int(row["memory_total_mib"]),
                "samples": 0,
                "min_memory_used_mib": int(row["memory_used_mib"]),
                "max_memory_used_mib": int(row["memory_used_mib"]),
            },
        )
        current["samples"] += 1
        current["min_memory_used_mib"] = min(
            int(current["min_memory_used_mib"]), int(row["memory_used_mib"])
        )
        current["max_memory_used_mib"] = max(
            int(current["max_memory_used_mib"]), int(row["memory_used_mib"])
        )

    summary = {
        "name": args.name,
        "command": command,
        "returncode": returncode,
        "interrupted": interrupted,
        "started_at_utc": start_iso,
        "finished_at_utc": end_iso,
        "elapsed_wall_seconds": elapsed,
        "child_user_seconds": usage_after.ru_utime - usage_before.ru_utime,
        "child_system_seconds": usage_after.ru_stime - usage_before.ru_stime,
        "child_max_rss_kib": usage_after.ru_maxrss,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "gpu_memory_csv": str(gpu_path) if gpu_rows else None,
        "gpu_memory": sorted(by_gpu.values(), key=lambda item: int(item["gpu_index"])),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
