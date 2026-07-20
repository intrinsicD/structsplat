#!/usr/bin/env python3
"""Build the two-cell COMP-012 visual-comparison assets and metric manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve()
EVIDENCE_ROOT = SCRIPT.parent
REPO_ROOT = EVIDENCE_ROOT.parents[2]
ARCHIVED_VISUAL = EVIDENCE_ROOT / "visual"
STREAM_ROOT = EVIDENCE_ROOT / "streams"
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "results/comp011_ssp2v_actual_dev_v2r36_2026-07-17"
DEFAULT_OUTPUT = EVIDENCE_ROOT / "visual-rebuild"
SCRIPT_SHA_PLACEHOLDER = "resolved_at_runtime"
REPETITIONS = 1

CELLS: dict[str, dict[str, Any]] = {
    "jason": {
        "image_id": "jason-briscoe-149782",
        "display_name": "Jason Briscoe",
        "width": 768,
        "height": 512,
        "target": ARCHIVED_VISUAL / "jason-target.png",
        "target_sha256": ("28bfbd8a839cb1d2945ebe5fe8fc63d6545e3467a0b9cf3f5c18be15856b2523"),
        "baseline": STREAM_ROOT / "jason-baseline.ssp2e",
        "baseline_sha256": ("11f695461712e21eee5a5f7a9f8c3611d8d64a80766046905aef74aa5212e799"),
        "candidate": STREAM_ROOT / "jason-candidate.ssp2f",
        "candidate_sha256": ("3a904052197bdf0b7291fab82707f5ff5b8c6cd8aa3b43a636535544e84b8c6e"),
        "complete_bytes": 51549,
        "historical_conservative_delta": {
            "psnr": 0.931405480,
            "ms_ssim": 0.000196278,
            "lpips": -0.000448857,
        },
    },
    "nomao": {
        "image_id": "nomao-saeki-33553",
        "display_name": "Nomao Saeki",
        "width": 768,
        "height": 512,
        "target": ARCHIVED_VISUAL / "nomao-target.png",
        "target_sha256": ("b5db359139833a7fc4913122b94529b651ded1f6679b55e98614a7f5d57b9e22"),
        "baseline": STREAM_ROOT / "nomao-baseline.ssp2e",
        "baseline_sha256": ("0fb494bbae298e695fca45fa16fd900f0e6bb2d6f1b84aa9151d7c2662743c52"),
        "candidate": STREAM_ROOT / "nomao-candidate.ssp2f",
        "candidate_sha256": ("73ce43478766b34c4c81c6facc565765df0369cd389b6441717308c2593fe6cd"),
        "complete_bytes": 52223,
        "historical_conservative_delta": {
            "psnr": 0.860618088,
            "ms_ssim": 0.000171006,
            "lpips": -0.002742201,
        },
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def archive_path(path: Path) -> str:
    return path.resolve().relative_to(EVIDENCE_ROOT).as_posix()


def _verify_inputs() -> None:
    for cell in CELLS.values():
        for key in ("target", "baseline", "candidate"):
            path = Path(cell[key])
            if not path.is_file():
                raise RuntimeError(f"missing {key}: {path}")
            expected = str(cell[f"{key}_sha256"])
            observed = sha256_path(path)
            if observed != expected:
                raise RuntimeError(f"{key} hash mismatch for {path}: {observed} != {expected}")
        if Path(cell["baseline"]).stat().st_size != int(cell["complete_bytes"]):
            raise RuntimeError("baseline byte count differs from frozen cap")
        if Path(cell["candidate"]).stat().st_size != int(cell["complete_bytes"]):
            raise RuntimeError("candidate byte count differs from frozen cap")


def _decode_and_measure(
    *,
    runtime_root: Path,
    blob_path: Path,
    target_path: Path,
    output_png: Path | None,
) -> dict[str, Any]:
    source_root = runtime_root / "runtime/source"
    metric_site = runtime_root / "runtime/metrics/site"
    sys.path[0:0] = [
        os.fspath(metric_site),
        os.fspath(source_root),
        os.fspath(source_root / "src"),
    ]
    import torch

    from benchmarks import ssp2e_arithmetic as arithmetic
    from benchmarks import ssp2e_container as container
    from benchmarks import ssp2e_execute as execute
    from benchmarks import ssp2v_execute as vexecute
    from benchmarks import ssp2v_quality as quality

    renderer_record = json.loads((runtime_root / "runtime/renderer/record.json").read_text())
    quality.load_persisted_renderer(
        runtime_root,
        renderer_record["relative_path"],
        renderer_record["sha256"],
        renderer_record["bytes"],
        renderer_record["elf_identity"],
    )
    native = arithmetic.NativeArithmeticCoder(runtime_root / "runtime/native/libssp2e_range_v1.so")
    blob = blob_path.read_bytes()
    frequency_provider = None if blob[:5] == b"SSP2F" else vexecute._comp009_frequency_provider()
    decoded = container.decode_container(
        blob,
        arithmetic_decode=native.decode,
        arithmetic_encode=native.encode,
        frequency_provider=frequency_provider,
    )
    header = decoded.parsed.header
    arrays = execute.boundary_arrays(header, decoded.symbols)
    candidate = quality.render_boundary_once(header, arrays)
    target_u8 = np.ascontiguousarray(Image.open(target_path).convert("RGB"), dtype=np.uint8)
    target = quality.strict_target_tensor(target_u8)
    values, lpips_state = quality.exact_metric_values(candidate, target)

    png_record = None
    if output_png is not None:
        image = (
            candidate.detach()
            .to(device="cpu", dtype=torch.float32)
            .squeeze(0)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        image_u8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        if output_png.exists():
            raise RuntimeError(f"refusing to overwrite visual asset: {output_png}")
        Image.fromarray(image_u8, mode="RGB").save(output_png, format="PNG", compress_level=9)
        png_record = {
            "relative_path": output_png.name,
            "sha256": sha256_path(output_png),
            "bytes": output_png.stat().st_size,
            "mode": "RGB",
            "size": [int(header.width), int(header.height)],
            "conversion": "rint(clamp(float32_render,0,1)*255)_to_uint8",
        }

    return {
        "blob": {
            "path": blob_path.name,
            "sha256": sha256_bytes(blob),
            "complete_bytes": len(blob),
        },
        "boundary_state_sha256": execute._array_state_digest(arrays),
        "candidate_tensor": quality.tensor_record(candidate),
        "target_tensor": quality.tensor_record(target),
        "metrics": values.as_dict(),
        "lpips_state_sha256": lpips_state,
        "renderer_sha256": renderer_record["sha256"],
        "png": png_record,
    }


def _worker(args: argparse.Namespace) -> None:
    record = _decode_and_measure(
        runtime_root=args.runtime_root,
        blob_path=args.blob,
        target_path=args.target,
        output_png=args.output_png,
    )
    print(json.dumps(record, sort_keys=True))


def _worker_environment(runtime_root: Path) -> dict[str, str]:
    source_root = runtime_root / "runtime/source"
    metric_site = runtime_root / "runtime/metrics/site"
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "NO_PROXY": "*",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HOME": os.fspath(runtime_root / "runtime/isolated-home"),
            "PYTHONPATH": os.pathsep.join(
                (
                    os.fspath(metric_site),
                    os.fspath(source_root),
                    os.fspath(source_root / "src"),
                )
            ),
            "TORCH_HOME": os.fspath(runtime_root / "runtime/metrics/torch"),
            "STRUCTSPLAT_SOURCE_MANIFEST_SHA256": (
                "6f6a564d9a71e8db2c19ee60a1f959d2c8e517db07863820dad1eb66e741770e"
            ),
        }
    )
    return environment


def _run_worker(
    *,
    runtime_root: Path,
    blob: Path,
    target: Path,
    output_png: Path | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        os.fspath(Path(__file__).resolve()),
        "--worker",
        "--runtime-root",
        os.fspath(runtime_root),
        "--blob",
        os.fspath(blob),
        "--target",
        os.fspath(target),
    ]
    if output_png is not None:
        command.extend(["--output-png", os.fspath(output_png)])
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_worker_environment(runtime_root),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"metric worker failed with return code {completed.returncode}:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"metric worker emitted {len(lines)} stdout lines")
    return json.loads(lines[0])


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("psnr", "ms_ssim", "lpips")
    output: dict[str, Any] = {"repetitions": len(rows)}
    for metric in metrics:
        values = [float(row["metrics"][metric]) for row in rows]
        output[metric] = {
            "values": values,
            "mean": float(np.mean(values, dtype=np.float64)),
            "min": min(values),
            "max": max(values),
            "spread": max(values) - min(values),
        }
    return output


def _verify_runtime(runtime_root: Path) -> None:
    required = (
        runtime_root / "runtime/source",
        runtime_root / "runtime/metrics/site",
        runtime_root / "runtime/metrics/torch",
        runtime_root / "runtime/renderer/record.json",
        runtime_root / "runtime/native/libssp2e_range_v1.so",
    )
    missing = [os.fspath(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "the frozen COMP-011 metric runtime is incomplete; supply it with "
            f"--runtime-root (missing: {missing})"
        )


def _verify_archive() -> None:
    _verify_inputs()
    manifest_path = ARCHIVED_VISUAL / "metrics.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["script"]["sha256"] != sha256_path(SCRIPT):
        raise RuntimeError("archived manifest does not bind the current build script")
    for slug, cell in CELLS.items():
        archived = manifest["cells"][slug]
        if archived["target"]["sha256"] != str(cell["target_sha256"]):
            raise RuntimeError(f"{slug} target hash differs from the frozen manifest")
        for method in ("baseline", "candidate"):
            record = archived["methods"][method]
            if record["blob_sha256"] != str(cell[f"{method}_sha256"]):
                raise RuntimeError(f"{slug} {method} stream hash differs from the frozen manifest")
            visual_path = ARCHIVED_VISUAL / record["visual"]["relative_path"]
            if sha256_path(visual_path) != record["visual"]["sha256"]:
                raise RuntimeError(f"{slug} {method} PNG hash mismatch")
    print(
        json.dumps(
            {
                "status": "ok",
                "metrics_json_sha256": sha256_path(manifest_path),
                "script_sha256": sha256_path(SCRIPT),
                "cells": sorted(CELLS),
            },
            sort_keys=True,
        )
    )


def _build(*, runtime_root: Path, output: Path) -> None:
    _verify_inputs()
    _verify_runtime(runtime_root)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    script_sha256 = sha256_path(SCRIPT)
    manifest: dict[str, Any] = {
        "schema": "structsplat.comp012.visual-comparison.v1",
        "archive_path_base": "evidence_bundle_root",
        "scope": "already_exposed_development_calibration_only",
        "bit_tuple": [12, 6, 6, 8],
        "repetitions_per_fitted_image": REPETITIONS,
        "metric_names": ["psnr", "ms_ssim", "lpips"],
        "metric_direction": {
            "psnr": "higher",
            "ms_ssim": "higher",
            "lpips": "lower",
        },
        "script": {
            "path": archive_path(SCRIPT),
            "sha256": script_sha256,
            "placeholder": SCRIPT_SHA_PLACEHOLDER,
        },
        "runtime_dependency": {
            "kind": "frozen_comp011_metric_runtime",
            "provided_via": "--runtime-root",
            "persisted_path": False,
        },
        "cells": {},
        "claim_boundary": (
            "Visual report for two already-exposed calibration cells. "
            "No held-out, convergence, performance, compression-isolation, "
            "expressiveness, default, or state-of-the-art claim."
        ),
    }
    for slug, cell in CELLS.items():
        target_output = output / f"{slug}-target.png"
        shutil.copyfile(Path(cell["target"]), target_output)
        target_image = Image.open(target_output).convert("RGB")
        if target_image.size != (int(cell["width"]), int(cell["height"])):
            raise RuntimeError("target dimensions differ from frozen cell dimensions")
        target_record = {
            "source_path": archive_path(Path(cell["target"])),
            "source_sha256": str(cell["target_sha256"]),
            "relative_path": target_output.name,
            "sha256": sha256_path(target_output),
            "bytes": target_output.stat().st_size,
            "mode": "RGB",
            "size": [int(cell["width"]), int(cell["height"])],
        }

        methods: dict[str, Any] = {}
        for method in ("baseline", "candidate"):
            rows = []
            output_png = output / f"{slug}-{method}.png"
            for repetition in range(REPETITIONS):
                row = _run_worker(
                    runtime_root=runtime_root,
                    blob=Path(cell[method]),
                    target=Path(cell["target"]),
                    output_png=output_png if repetition == 0 else None,
                )
                row["blob"]["path"] = archive_path(Path(cell[method]))
                row["repetition"] = repetition + 1
                rows.append(row)
            if rows[0]["png"] is None:
                raise RuntimeError("first repetition did not persist its PNG")
            for row in rows:
                if row["blob"]["sha256"] != str(cell[f"{method}_sha256"]):
                    raise RuntimeError("worker blob hash differs from frozen input")
                if row["blob"]["complete_bytes"] != int(cell["complete_bytes"]):
                    raise RuntimeError("worker blob bytes differ from frozen cap")
            methods[method] = {
                "format": "ssp2e" if method == "baseline" else "ssp2f",
                "archived_relative_path": archive_path(Path(cell[method])),
                "blob_sha256": str(cell[f"{method}_sha256"]),
                "complete_bytes": int(cell["complete_bytes"]),
                "bpp": (
                    8.0 * int(cell["complete_bytes"]) / (int(cell["width"]) * int(cell["height"]))
                ),
                "visual": rows[0]["png"],
                "metric_aggregate": _aggregate(rows),
                "metric_rows": rows,
            }

        baseline = methods["baseline"]["metric_aggregate"]
        candidate = methods["candidate"]["metric_aggregate"]
        mean_delta = {
            metric: (float(candidate[metric]["mean"]) - float(baseline[metric]["mean"]))
            for metric in ("psnr", "ms_ssim", "lpips")
        }
        conservative_delta = {
            "psnr": float(candidate["psnr"]["min"]) - float(baseline["psnr"]["max"]),
            "ms_ssim": float(candidate["ms_ssim"]["min"]) - float(baseline["ms_ssim"]["max"]),
            "lpips": float(candidate["lpips"]["max"]) - float(baseline["lpips"]["min"]),
        }
        manifest["cells"][slug] = {
            "image_id": str(cell["image_id"]),
            "display_name": str(cell["display_name"]),
            "width": int(cell["width"]),
            "height": int(cell["height"]),
            "pixels": int(cell["width"]) * int(cell["height"]),
            "target": target_record,
            "methods": methods,
            "candidate_minus_baseline_mean": mean_delta,
            "candidate_minus_baseline_conservative": conservative_delta,
            "historical_three_repeat_conservative_delta": dict(
                cell["historical_conservative_delta"]
            ),
        }

    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output / "metrics.json").write_text(encoded)
    print(
        json.dumps(
            {
                "output": os.fspath(output),
                "metrics_json_sha256": sha256_path(output / "metrics.json"),
                "cells": {
                    slug: {
                        "mean_delta": value["candidate_minus_baseline_mean"],
                        "conservative_delta": value["candidate_minus_baseline_conservative"],
                    }
                    for slug, value in manifest["cells"].items()
                },
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--worker", action="store_true")
    actions.add_argument("--build", action="store_true")
    actions.add_argument("--verify", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blob", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--output-png", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.blob is None or args.target is None:
            parser.error("--worker requires --blob and --target")
        _worker(args)
    elif args.build:
        if args.blob is not None or args.target is not None or args.output_png is not None:
            parser.error("--build does not accept worker paths")
        _build(runtime_root=args.runtime_root.resolve(), output=args.output_dir.resolve())
    else:
        if args.blob is not None or args.target is not None or args.output_png is not None:
            parser.error("--verify does not accept worker paths")
        _verify_archive()


if __name__ == "__main__":
    main()
