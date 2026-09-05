#!/usr/bin/env python3
"""HIER-034 prospective shared-GPU correctness profile; speed eligibility is always false."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.experiments import hier034_basis_cache as original  # noqa: E402
from benchmarks.hier_research_report import (  # noqa: E402
    ResearchBundle, protocol_digest, sha256, write_json,
)

SOURCES = ["scripts/experiments/hier034_shared_cache.py", *original.SOURCES]
PROTOCOL = {
    **original.PROTOCOL, "version": 2, "execution_profile": "shared_correctness",
    "original_timing_source": "4b2c79f2a97e0bde5109d11ab717edd5881a1ca1",
    "timing_eligible": False, "performance_disposition": "inconclusive—shared-GPU contention",
    "decision": "six-pair numerical/checkpoint interchangeability only; every speed gate is false",
    "occupancy": "before/after every worker plus parent-process sampling throughout the matrix",
    "occupancy_interval_seconds": 1.0, "occupancy_timeout_seconds": 5,
    "resource_failures": "retain shared-resource OOM/timeouts; no inference of intrinsic cache infeasibility",
}


def gpu_snapshot():
    snapshot = {"wall_time_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_ns": time.monotonic_ns()}
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
            timeout=PROTOCOL["occupancy_timeout_seconds"])
        snapshot.update(status="ok", processes=output.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        snapshot.update(status="error", error=str(exc))
    return snapshot


@contextmanager
def monitor_gpu(path):
    """External to workers; stop/join before the immutable artifact manifest is sealed."""
    stopped = threading.Event()
    with Path(path).open("x") as stream:
        def record(phase):
            stream.write(json.dumps({"phase": phase, **gpu_snapshot()}, sort_keys=True) + "\n")
            stream.flush()
        record("start")
        def sample():
            while not stopped.wait(PROTOCOL["occupancy_interval_seconds"]):
                record("sample")
        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        try:
            yield
        finally:
            stopped.set()
            monitor.join()
            record("end")


def worker(request_path):
    directory = Path(request_path).parent
    write_json(directory / "occupancy_before.json", gpu_snapshot())
    try:
        original.worker(request_path)
    finally:
        write_json(directory / "occupancy_after.json", gpu_snapshot())
    config = json.loads((directory / "config.json").read_text())
    config.update(execution_profile=PROTOCOL["execution_profile"], timing_eligible=False,
                  occupancy_before_sha256=sha256(directory / "occupancy_before.json"),
                  occupancy_after_sha256=sha256(directory / "occupancy_after.json"))
    write_json(directory / "config.json", config)
    row = json.loads((directory / "row.json").read_text())
    row.update(execution_profile=PROTOCOL["execution_profile"], timing_eligible=False)
    for name in ("occupancy_before.json", "occupancy_after.json"):
        row["artifacts"][name] = f"cells/{row['cell_id']}/{name}"
    write_json(directory / "row.json", row)


def analyze_bundle(root, rows):
    # Original numerical gates are retained; the timing gate cannot pass under this profile.
    return original.analyze_bundle(root, rows, timing_eligible=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?")
    parser.add_argument("--base-bundle")
    parser.add_argument("--worker")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--print-protocol-digest", action="store_true")
    parser.add_argument("--approved-protocol-digest")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    digest = protocol_digest(PROTOCOL, SOURCES)
    if args.print_protocol_digest:
        print(digest)
        return
    if not args.outdir or (not args.smoke and
            (args.approved_protocol_digest != digest or not args.base_bundle)):
        parser.error("formal run requires outdir, base bundle and exact approved digest")
    workloads = [(family, seed) for family in PROTOCOL["families"] for seed in PROTOCOL["seeds"]]
    workloads += [("hier031", 0)]
    orders = list(itertools.permutations(PROTOCOL["backends"]))
    if args.smoke:
        workloads, orders = [("smooth", 77)], [tuple(PROTOCOL["backends"])]
    requests = [{"family": family, "seed": seed, "repeat": repeat, "backend": backend,
                 "cell_id": f"{family}_s{seed}_r{repeat}_{backend}",
                 "base_bundle": args.base_bundle, "smoke": args.smoke}
                for family, seed in workloads for repeat, order in enumerate(orders) for backend in order]
    bundle = ResearchBundle(args.outdir, task="HIER-034", protocol=PROTOCOL, digest=digest,
                            expected_cells=[r["cell_id"] for r in requests], diagnostic=args.smoke,
                            source_paths=SOURCES)
    rows = []
    with monitor_gpu(bundle.root / "gpu_occupancy.jsonl"):
        for request in requests:
            directory = bundle.root / "cells" / request["cell_id"]
            directory.mkdir(parents=True)
            request_path = directory / "request.json"
            write_json(request_path, request)
            try:
                with (directory / "worker.log").open("w") as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path)],
                                   check=True, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=PROTOCOL["worker_timeout_seconds"])
                row = json.loads((directory / "row.json").read_text())
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                row = {**request, "status": "error", "error": str(exc),
                       "execution_profile": PROTOCOL["execution_profile"], "timing_eligible": False,
                       "artifacts": {"worker.log": f"cells/{request['cell_id']}/worker.log"}}
                for name in ("occupancy_before.json", "occupancy_after.json"):
                    if (directory / name).is_file():
                        row["artifacts"][name] = f"cells/{request['cell_id']}/{name}"
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("cell_id", "status", "total_seconds") if k in row}), flush=True)
    analyze_bundle(bundle.root, rows)
    for row in rows:
        row.setdefault("artifacts", {}).update(decision="decision.json", gpu_occupancy="gpu_occupancy.jsonl")
    bundle.finish(rows, title="HIER-034 — Shared-GPU cache correctness",
                  interpretation="Same180-cell cache matrix and numerical/checkpoint gates; shared-GPU timing is ineligible for every speed claim. Elapsed times and ratios are descriptive. Resource failures remain visible and do not prove intrinsic cache infeasibility.")


if __name__ == "__main__":
    main()
