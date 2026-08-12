#!/usr/bin/env python3
"""Finalize HIER-031 HTML/crops against its frozen black-matted objective arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier022_additive_continuation as h22  # noqa: E402
from scripts.experiments import hier031_exact7k_masked_boundary_detail as h31  # noqa: E402


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _protected_hashes(root: Path, rows: list[dict[str, object]]) -> dict[str, str]:
    paths = [
        root / "metrics.json",
        root / "metrics.jsonl",
        root / "metrics.csv",
        root / "decision.json",
        root / "attempts.json",
        root / "feasibility.json",
        root / "feasibility.npz",
    ]
    paths.extend(root / str(row["artifact_dir"]) / "field.gaussian.npz" for row in rows)
    return {
        str(path.relative_to(root)): h22.report_utils._sha256(path)
        for path in paths
    }


def finalize(root: Path) -> dict[str, object]:
    if not (root / "COMPLETED").is_file():
        raise ValueError("HIER-031 presentation finalization requires a completed bundle")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("schema") != h31.REPORT_SCHEMA:
        raise ValueError(f"unexpected report schema {metrics.get('schema')!r}")
    rows = metrics.get("rows")
    if not isinstance(rows, list) or len(rows) != len(h31.ARMS):
        raise ValueError("HIER-031 metrics do not contain the complete frozen arm matrix")

    protected_before = _protected_hashes(root, rows)
    index_before = h22.report_utils._sha256(root / "index.html")
    manifest_before = h22.report_utils._sha256(root / "manifest.json")
    changed: list[str] = []
    for row in rows:
        artifact_dir = root / str(row["artifact_dir"])
        with np.load(artifact_dir / "analysis.npz", allow_pickle=False) as analysis:
            worst_bounds = tuple(int(value) for value in analysis["worst_crop_bounds"])
            hair_bounds = tuple(int(value) for value in analysis["hair_crop_bounds"])
        target = _load_rgb(artifact_dir / "objective_source.png")
        reconstruction = _load_rgb(artifact_dir / "objective_reconstruction.png")
        error = _load_rgb(artifact_dir / "objective_error.png")
        outputs = (
            ("source_crop.png", target, worst_bounds),
            ("reconstruction_crop.png", reconstruction, worst_bounds),
            ("error_crop.png", error, worst_bounds),
            ("hair_source_crop.png", target, hair_bounds),
            ("hair_reconstruction_crop.png", reconstruction, hair_bounds),
            ("hair_error_crop.png", error, hair_bounds),
        )
        for name, image, bounds in outputs:
            h22.viz_utils._save_crop(artifact_dir / name, image, bounds)
            changed.append(str((artifact_dir / name).relative_to(root)))

    record_path = root / "presentation_finalization.json"
    h31._write_json(
        record_path,
        {
            "schema": h31.REPORT_SCHEMA,
            "status": "presentation_finalization_in_progress",
        },
    )
    attempts = json.loads((root / "attempts.json").read_text(encoding="utf-8"))["attempts"]
    feasibility = json.loads((root / "feasibility.json").read_text(encoding="utf-8"))
    decision = json.loads((root / "decision.json").read_text(encoding="utf-8"))
    h31._write_report(root, rows, attempts, feasibility, decision)

    protected_after = _protected_hashes(root, rows)
    if protected_after != protected_before:
        raise RuntimeError("presentation finalization changed a protected measurement artifact")
    record: dict[str, object] = {
        "schema": h31.REPORT_SCHEMA,
        "status": "presentation_finalized",
        "reason": (
            "the first HTML serializer showed full-source rather than black-matted objective "
            "error views; measurements and fields were already complete"
        ),
        "measurement_recomputed": False,
        "fields_recomputed": False,
        "metrics_recomputed": False,
        "index_sha256_before": index_before,
        "index_sha256_after": h22.report_utils._sha256(root / "index.html"),
        "manifest_sha256_before": manifest_before,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_unchanged": True,
        "changed_presentation_files": sorted(changed + ["index.html"]),
        "producer_sha256": h22.report_utils._sha256(
            ROOT / "scripts/experiments/hier031_exact7k_masked_boundary_detail.py"
        ),
        "finalizer_sha256": h22.report_utils._sha256(Path(__file__)),
    }
    h31._write_json(record_path, record)
    h31._write_manifest(root)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.bundle), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
