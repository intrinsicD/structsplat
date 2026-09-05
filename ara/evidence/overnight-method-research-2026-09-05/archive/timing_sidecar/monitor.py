"""Read-only occupancy sidecar for the already frozen HIER-034 timing assay."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

OUT = Path('/home/alex/Documents/structsplat/results/hier034_timing_occupancy_2026-09-05')
SOURCE = Path('/tmp/structsplat-overnight-VmVXmg/source')
COMMAND = [sys.executable, 'scripts/experiments/hier034_basis_cache.py',
    '/home/alex/Documents/structsplat/results/hier034_basis_cache_2026-09-05',
    '--base-bundle', '/home/alex/Documents/structsplat/results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_2026-08-12',
    '--approved-protocol-digest', 'a5be4997c39a13d59eada1962a67837ff447bf9145e44b5dcca3eaaaacb66436']

def snapshot(driver_pid=None):
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'monotonic_ns': time.monotonic_ns()}
    try:
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=5)
        processes = []
        for line in query.stdout.splitlines():
            pid, memory = (int(value.strip()) for value in line.split(','))
            ancestry = []
            current = pid
            try:
                while current > 1 and len(ancestry) < 20:
                    ancestry.append(current)
                    current = int(Path(f'/proc/{current}/stat').read_text().rsplit(')', 1)[1].split()[1])
                owned = driver_pid in ancestry if driver_pid is not None else False
            except (OSError, ValueError):
                owned = None
            processes.append({'pid': pid, 'memory_mib': memory, 'owned_by_driver': owned})
        result.update(status='ok', processes=processes)
    except (OSError, subprocess.SubprocessError) as exc:
        result.update(status='error', error=str(exc))
    return result

preflight = snapshot()
if preflight['status'] != 'ok' or preflight['processes']:
    raise SystemExit(f'GPU not idle; timing assay not started: {preflight}')
OUT.mkdir(exist_ok=False)
source_bytes = Path(__file__).read_bytes()
(OUT / 'monitor.py').write_bytes(source_bytes)
metadata = {'command': COMMAND, 'source_worktree': str(SOURCE), 'monitor_sha256': hashlib.sha256(source_bytes).hexdigest(),
    'sampling_interval_seconds': 1, 'sampling_timeout_seconds': 5,
    'scope': 'External observational sidecar; scientific source and frozen complete matrix unchanged.',
    'limitations': 'Point samples do not prove continuous exclusivity; no CPU occupancy inference.'}
(OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
with (OUT / 'occupancy.jsonl').open('x') as stream:
    stream.write(json.dumps({'phase': 'preflight', **preflight}) + '\n')
    stream.flush()
    driver = subprocess.Popen(COMMAND, cwd=SOURCE)
    stop = threading.Event()
    def record(phase):
        stream.write(json.dumps({'phase': phase, 'driver_pid': driver.pid, **snapshot(driver.pid)}) + '\n')
        stream.flush()
    record('start')
    def monitor():
        while not stop.wait(1):
            record('sample')
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        returncode = driver.wait()
    finally:
        stop.set()
        thread.join()
        record('end')
(OUT / 'completion.json').write_text(json.dumps({'returncode': returncode, 'driver_pid': driver.pid,
    'utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
raise SystemExit(returncode)
