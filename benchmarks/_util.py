"""Shared benchmark plumbing (BENCH-002): reproducible config records and empty-safe stats.

Every harness must persist enough to reproduce its numbers from its own outdir (invariant 5):
the resolved arguments, the device, and the package/toolchain versions. It must also survive a
single broken arm (e.g. renderer=cuda on a CPU box) — summary writers use ``mean_or_none`` so a
method with zero ok rows yields ``None`` rather than raising ``StatisticsError``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
import subprocess


def package_versions() -> dict:
    """Versions that move the numbers: torch (+CUDA), numpy, and structsplat itself."""
    out = {}
    try:
        import torch
        out["torch"] = torch.__version__
        out["torch_cuda"] = getattr(torch.version, "cuda", None)
        out["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        out["torch"] = None
    try:
        import numpy as np
        out["numpy"] = np.__version__
    except Exception:
        out["numpy"] = None
    try:
        import importlib.metadata as _m
        out["structsplat"] = _m.version("structsplat")
    except Exception:
        try:
            import structsplat
            out["structsplat"] = getattr(structsplat, "__version__", None)
        except Exception:
            out["structsplat"] = None
    return out


def repository_state() -> dict:
    """Record source provenance without embedding the potentially large working diff."""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        tracked_diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return {
            "root": str(root),
            "commit": None,
            "dirty": None,
            "tracked_diff_sha256": None,
            "untracked_files": [],
        }
    untracked = [
        line[3:] for line in status.splitlines()
        if line.startswith("?? ")
    ]
    return {
        "root": str(root),
        "commit": commit,
        "dirty": bool(status.strip()),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_files": untracked,
    }


def run_config(resolved: dict, device: str | None = None, extra: dict | None = None) -> dict:
    """Assemble a self-contained run record: resolved args + device + versions.

    GPU renders are nondeterministic (atomicAdd / CUDA index_add), so the exact device and
    renderer travel with the results and reproducibility is bit-exact only on CPU (documented in
    the benchmark skill / README).
    """
    cfg = {
        "resolved": _jsonable(resolved),
        "device": device,
        "versions": package_versions(),
        "repository": repository_state(),
    }
    if extra:
        cfg.update(_jsonable(extra))
    return cfg


def write_config(outdir: str, config: dict, name: str = "config.json") -> None:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, name), "w") as f:
        json.dump(config, f, indent=2, default=str)


def mean_or_none(vals):
    """Mean of a possibly-empty iterable; None instead of raising on empty (BENCH-002)."""
    vals = _clean_numbers(vals)
    return mean(vals) if vals else None


def mean_std_or_none(vals):
    vals = _clean_numbers(vals)
    if not vals:
        return None, None
    return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)


def add_seed_args(parser) -> None:
    """Add the benchmark seed axis while preserving the legacy single-seed flag."""
    parser.add_argument("--seed", type=int, default=None,
                        help="legacy single-seed alias; ignored when --seeds is supplied")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="random seeds to sweep; reports aggregate image x seed statistics")


def resolve_seeds(seed: int | None = None, seeds: list[int] | tuple[int, ...] | None = None) -> list[int]:
    """Resolve --seed/--seeds into a non-empty list; --seeds wins when both are supplied."""
    resolved = list(seeds) if seeds is not None else ([seed] if seed is not None else [0])
    if not resolved:
        raise ValueError("at least one seed is required")
    return [int(s) for s in resolved]


def fmt_mean_std(vals, digits: int = 4) -> tuple[str, str]:
    """Return formatted mean and population std cells for markdown/CSV summaries."""
    m, s = mean_std_or_none(vals)
    if m is None:
        return "-", "-"
    return f"{m:.{digits}f}", f"{s:.{digits}f}"


def _jsonable(obj):
    """Best-effort convert argparse Namespaces / Paths / dataclasses to plain JSON types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(obj).items()}
    return str(obj)


def _clean_numbers(vals):
    out = []
    for v in vals:
        if v is None:
            continue
        fv = float(v)
        if math.isnan(fv):
            continue
        out.append(fv)
    return out
