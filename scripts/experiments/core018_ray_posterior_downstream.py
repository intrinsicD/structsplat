#!/usr/bin/env python3
"""Run CORE-018's disjoint codec-native ray-posterior diagnostic.

The driver creates one immutable packet set from construction views, cold-reloads those exact
bytes for every arm, and compares ordinary CompactCarve interior consensus with packet-feature
ray posteriors before and after reciprocal consistency.  Source RGB files are used only for packet
construction and the shared realtime-gs optimization targets; posterior geometry reads only cold
packet queries, packet-derived features, construction cameras, and explicit bounds.

Reproduce from the StructSplat root with::

    PYTHONPATH=src:/home/alex/Documents/realtime-gs/src \
      /home/alex/Documents/realtime-gs/.venv/bin/python \
      scripts/experiments/core018_ray_posterior_downstream.py \
      --out results/core018_ray_posterior_karate_frame00060_2026-08-06_v1

The output directory is immutable: this script refuses to overwrite it.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gc
import hashlib
from html import escape
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch

from rtgs.core.metrics import image_metrics
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import (
    CompactCarveConfig,
    CompactCarveInitializer,
    make_placement_progress_printer,
)
from rtgs.lift.surfel_init import SurfelInitConfig, reconcile_covariances
from rtgs.optim.density import DensityConfig
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

from structsplat.codec_native_field import (
    CodecNativeField,
    CodecNativeFieldConfig,
    build_codec_native_field,
)
from structsplat.metrics import LPIPS, ms_ssim
from structsplat.realtime_gs_adapter import make_realtime_gs_view
from structsplat.realtime_gs_ray_posterior import (
    RayPosteriorConfig,
    build_packet_feature_pyramids,
    initialize_occlusion_aware_ray_posterior,
)


STRUCTSPLAT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RTGS_ROOT = Path("/home/alex/Documents/realtime-gs")
DEFAULT_FRAME = Path("/home/alex/Dropbox/Work/Janelle/karate/frame_00060")
ALL_CAMERA_IDS = (
    "C0000",
    "C0001",
    "C0004",
    "C0005",
    "C0006",
    "C0007",
    "C0008",
    "C0010",
    "C0012",
    "C0013",
    "C0014",
    "C0016",
    "C0018",
    "C0020",
    "C0021",
    "C0022",
    "C0024",
    "C0025",
    "C0026",
    "C0028",
    "C0029",
    "C0030",
    "C0031",
    "C0034",
    "C0037",
    "C0038",
    "C0039",
    "C1000",
    "C1001",
    "C1002",
    "C1004",
    "C1005",
)
REPORT_CAMERA_IDS = ("C0004", "C0025", "C1004", "C1005")
TRAIN_CAMERA_IDS = tuple(value for value in ALL_CAMERA_IDS if value not in REPORT_CAMERA_IDS)
ARMS = ("interior", "posterior_no_reciprocal", "posterior_reciprocal")
ARM_LABELS = {
    "interior": "ordinary interior consensus + surface cover",
    "posterior_no_reciprocal": "ray posterior + surface cover",
    "posterior_reciprocal": "ray posterior + reciprocal consistency + surface cover",
}
PACKET_DOWNSCALE = 4
EVALUATION_DOWNSCALE = 8
INITIAL_GAUSSIANS = 10_000
MAX_GAUSSIANS = 30_000
ITERATIONS = 1_500
FIXED_PREFIX_STEPS = 500
EVAL_EVERY = 100
SEED = 0
SOURCE_SNAPSHOTS = (
    "src/structsplat/codec_native_field.py",
    "src/structsplat/realtime_gs_adapter.py",
    "src/structsplat/realtime_gs_ray_posterior.py",
    "scripts/experiments/core018_ray_posterior_downstream.py",
)
RTGS_SNAPSHOTS = (
    "src/rtgs/lift/compact_carve.py",
    "src/rtgs/lift/surfel_init.py",
    "src/rtgs/optim/trainer.py",
    "src/rtgs/optim/density.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository_record(root: Path) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "root": str(root),
        "head": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "numpy", "Pillow", "matplotlib", "lpips", "pytorch-msssim"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _snapshot_sources(out: Path, rtgs_root: Path) -> list[dict[str, Any]]:
    records = []
    for repository, root, names in (
        ("structsplat", STRUCTSPLAT_ROOT, SOURCE_SNAPSHOTS),
        ("realtime_gs", rtgs_root, RTGS_SNAPSHOTS),
    ):
        for name in names:
            source = root / name
            target = out / "sources" / repository / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records.append(
                {
                    "repository": repository,
                    "source": str(source),
                    "snapshot": _artifact(target, out),
                }
            )
    return records


def _source_path(frame: Path, view_name: str) -> Path:
    candidates = [
        path
        for path in (frame / "rgb").iterdir()
        if path.stem.lower() == view_name.lower()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one original image for {view_name}, found {candidates}")
    return candidates[0]


def _save_rgb(path: Path, value: torch.Tensor | np.ndarray) -> None:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def _labeled(value: torch.Tensor | np.ndarray, label: str, header: int = 26) -> Image.Image:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels).convert("RGB")
    result = Image.new("RGB", (image.width, image.height + header), "white")
    result.paste(image, (0, header))
    ImageDraw.Draw(result).text((5, 6), label, fill="black")
    return result


def _sheet(rows: list[list[Image.Image]]) -> Image.Image:
    row_images = []
    for cells in rows:
        width = sum(cell.width for cell in cells)
        height = max(cell.height for cell in cells)
        row = Image.new("RGB", (width, height), "white")
        x = 0
        for cell in cells:
            row.paste(cell, (x, 0))
            x += cell.width
        row_images.append(row)
    result = Image.new(
        "RGB",
        (max(row.width for row in row_images), sum(row.height for row in row_images)),
        "white",
    )
    y = 0
    for row in row_images:
        result.paste(row, (0, y))
        y += row.height
    return result


def _carve_config() -> CompactCarveConfig:
    return CompactCarveConfig(
        n_init_3d=INITIAL_GAUSSIANS,
        candidate_multiplier=2,
        anchor_mode="mass_random",
        samples_per_ray=24,
        query_batch_size=4_096,
        seed=SEED,
        bounds_scale=0.5,
        min_views=2,
        hull_fraction=0.85,
        coverage_scale=1.0,
        coverage_threshold=0.40,
        color_std_sigma=0.20,
        min_score=0.05,
    )


def _posterior_config(apply_reciprocal: bool) -> RayPosteriorConfig:
    return RayPosteriorConfig(
        feature_model="dinov2_vits14",
        feature_max_side=518,
        feature_patch_size=14,
        feature_device="cuda",
        feature_storage_dtype="float16",
        target_views=4,
        target_baseline_deg=18.0,
        min_baseline_deg=3.0,
        max_baseline_deg=65.0,
        best_view_count=2,
        min_evidence_views=2,
        dustbin_cost=0.65,
        view_dispersion_weight=0.25,
        posterior_temperature=0.08,
        dino_weight=1.0,
        detail_weight=0.25,
        fine_samples=9,
        fine_half_width_steps=1.0,
        score_batch_rays=1_024,
        reciprocal_pixel_radius=18.0,
        reciprocal_depth_extent_fraction=0.04,
        reciprocal_world_extent_fraction=0.06,
        min_reciprocal_views=1,
        apply_reciprocal=apply_reciprocal,
        allow_confidence_fallback=True,
        min_primary_fraction=0.75 if apply_reciprocal else 1.0,
        apply_surface_cover=True,
    )


def _train_config() -> TrainConfig:
    return TrainConfig(
        iterations=ITERATIONS,
        rasterizer="gsplat",
        device="cuda",
        densify=True,
        density_strategy="classic",
        density=DensityConfig(
            start_iter=600,
            stop_iter=1_400,
            every=100,
            grad_threshold=8e-4,
            absgrad=True,
            split_scale_frac=0.01,
            split_factor=1.6,
            prune_opacity=0.005,
            prune_scale_frac=0.1,
            max_gaussians=MAX_GAUSSIANS,
            opacity_reset_every=1_000,
            opacity_reset_value=0.011,
            revised_opacity=True,
        ),
        eval_every=EVAL_EVERY,
        checkpoint_policy="final",
        target_sh_degree=3,
        sh_degree_interval=375,
        use_masks=False,
        random_background=False,
        packed=False,
        antialiased=True,
        record_train_metrics=False,
        validate_render_finite=True,
        seed=SEED,
    )


def _build_shared_packets(
    packet_scene: Any,
    frame: Path,
    out: Path,
) -> dict[str, Any]:
    root = out / "shared_packets"
    records = []
    started = time.perf_counter()
    for index, (view_name, image) in enumerate(
        zip(packet_scene.view_names, packet_scene.images, strict=True)
    ):
        source = _source_path(frame, view_name)
        source_payload = source.read_bytes()
        config = CodecNativeFieldConfig(
            appearance_codec="webp",
            appearance_quality=80,
            lattice_sigma_px=0.45,
            lattice_radius_px=3,
            lattice_prefilter_steps=8,
            structural_count=1_024,
            structural_seed=index,
        )
        encode_started = time.perf_counter()
        packet = build_codec_native_field(
            image.detach().cpu().numpy(),
            config=config,
            mask=np.ones(image.shape[:2], dtype=bool),
            source_payload=source_payload,
        )
        encode_seconds = time.perf_counter() - encode_started
        packet_path = root / view_name / f"{view_name}.sgdp"
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = packet.save(packet_path)
        decoded_error = np.abs(packet.decoded_appearance - image.detach().cpu().numpy())
        mse = float(np.square(decoded_error).mean())
        decoded_path = packet_path.parent / "decoded.png"
        error_path = packet_path.parent / "decoded_error_x8.png"
        _save_rgb(decoded_path, packet.decoded_appearance)
        _save_rgb(error_path, decoded_error * 8.0)
        records.append(
            {
                "view_name": view_name,
                "source": {
                    "path": str(source),
                    "bytes": len(source_payload),
                    "sha256": hashlib.sha256(source_payload).hexdigest(),
                },
                "packet_image_sha256": _tensor_sha256(image),
                "config": dataclasses.asdict(config),
                "packet": _artifact(packet_path, out),
                "packet_ledger": dataclasses.asdict(ledger),
                "decoded": _artifact(decoded_path, out),
                "decoded_error_x8": _artifact(error_path, out),
                "decoded_psnr_db": 120.0 if mse <= 1e-12 else -10.0 * math.log10(mse),
                "encode_seconds": encode_seconds,
            }
        )
    return {
        "schema": "core018.shared_packets.v1",
        "view_count": len(records),
        "complete_packet_bytes": sum(row["packet"]["bytes"] for row in records),
        "original_source_bytes": sum(row["source"]["bytes"] for row in records),
        "encode_seconds": sum(row["encode_seconds"] for row in records),
        "wall_seconds": time.perf_counter() - started,
        "views": records,
    }


def _reload_packets(
    packet_record: dict[str, Any],
    packet_scene: Any,
    out: Path,
) -> tuple[ReconstructionInputs, list[Any], dict[str, Any]]:
    if [row["view_name"] for row in packet_record["views"]] != list(packet_scene.view_names):
        raise ValueError("shared packet order does not match packet scene")
    views = []
    decode_seconds = 0.0
    adapter_seconds = 0.0
    for row in packet_record["views"]:
        started = time.perf_counter()
        packet = CodecNativeField.load(out / row["packet"]["path"])
        decode_seconds += time.perf_counter() - started
        started = time.perf_counter()
        view = make_realtime_gs_view(packet, device="cpu", query_device="cuda")
        torch.cuda.synchronize()
        adapter_seconds += time.perf_counter() - started
        views.append(view)
    inputs = ReconstructionInputs(
        observations=[view.structural_field for view in views],
        cameras=list(packet_scene.cameras),
        view_names=list(packet_scene.view_names),
        bounds_hint=(packet_scene.bounds_hint[0].clone(), packet_scene.bounds_hint[1]),
        name="core018-shared-codec-packets",
    )
    return inputs, views, {
        "complete_packet_bytes": packet_record["complete_packet_bytes"],
        "original_source_bytes": packet_record["original_source_bytes"],
        "cold_decode_seconds": decode_seconds,
        "adapter_seconds": adapter_seconds,
        "index_entries": sum(view.query_backend.n_entries for view in views),
        "index_payload_bytes": sum(view.query_backend.payload_bytes for view in views),
        "packet_hashes": [row["packet"]["sha256"] for row in packet_record["views"]],
    }


def _gradient_mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred_dx = prediction[:, 1:] - prediction[:, :-1]
    target_dx = target[:, 1:] - target[:, :-1]
    pred_dy = prediction[1:, :] - prediction[:-1, :]
    target_dy = target[1:, :] - target[:-1, :]
    return float(
        torch.cat(
            [(pred_dx - target_dx).abs().reshape(-1), (pred_dy - target_dy).abs().reshape(-1)]
        ).mean()
    )


def _evaluate_indices(scene: Any, model: Any, renderer: Any, indices: list[int]) -> dict[str, Any]:
    records = []
    with torch.no_grad():
        for index in indices:
            target = scene.images[index].to(model.means.device)
            camera = scene.cameras[index].to(model.means.device)
            output = renderer.render(model, camera)
            prediction = output.color.clamp(0.0, 1.0)
            values = image_metrics(prediction, target)
            values["ms_ssim"] = ms_ssim(prediction, target)
            try:
                values["lpips"] = LPIPS.distance(prediction, target)
            except Exception as error:
                values["lpips"] = None
                values["lpips_error"] = f"{type(error).__name__}: {error}"[:300]
            absolute = (prediction - target).abs()
            values.update(
                {
                    "mae": float(absolute.mean()),
                    "p95_abs": float(torch.quantile(absolute, 0.95)),
                    "p99_abs": float(torch.quantile(absolute, 0.99)),
                    "gradient_mae": _gradient_mae(prediction, target),
                    "alpha_mean": float(output.alpha.mean()),
                    "alpha_p05": float(torch.quantile(output.alpha, 0.05)),
                    "view_name": scene.view_names[index],
                }
            )
            records.append(values)
    numeric = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    aggregate = {}
    for key in numeric:
        values = [float(record[key]) for record in records if record.get(key) is not None]
        if values:
            aggregate[key] = float(np.mean(values))
    if any(record.get("lpips") is None for record in records):
        aggregate["lpips"] = None
    return {"aggregate": aggregate, "per_view": records}


def _evaluate_model(scene: Any, model: Any, renderer: Any) -> dict[str, Any]:
    return {
        "train": _evaluate_indices(scene, model, renderer, list(scene.training_views)),
        "reporting": _evaluate_indices(scene, model, renderer, list(scene.testing_views)),
    }


def _depth_rgb(output: Any) -> np.ndarray:
    alpha = output.alpha.detach()
    expected = output.depth.detach() / alpha.clamp_min(1e-6)
    valid = alpha > 0.05
    if bool(valid.any()):
        lo = torch.quantile(expected[valid], 0.05)
        hi = torch.quantile(expected[valid], 0.95)
        normalized = ((expected - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    else:
        normalized = torch.zeros_like(expected)
    rgb = plt.get_cmap("viridis")(normalized.cpu().numpy())[..., :3]
    rgb[~valid.cpu().numpy()] = 0.0
    return rgb.astype(np.float32)


def _save_visuals(
    arm: str,
    scene: Any,
    initial: Any,
    final: Any,
    renderer: Any,
    out: Path,
) -> dict[str, Any]:
    root = out / "arms" / arm / "visuals"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    views = {}
    initial = initial.to("cuda")
    final = final.to("cuda")
    with torch.no_grad():
        for index in scene.testing_views:
            name = scene.view_names[index]
            target = scene.images[index].cuda()
            camera = scene.cameras[index].to("cuda")
            initial_output = renderer.render(initial, camera)
            final_output = renderer.render(final, camera)
            initial_rgb = initial_output.color.clamp(0.0, 1.0)
            final_rgb = final_output.color.clamp(0.0, 1.0)
            error = (final_rgb - target).abs().mul(4.0).clamp(0.0, 1.0)
            alpha = final_output.alpha.clamp(0.0, 1.0)[..., None].expand(-1, -1, 3)
            depth = _depth_rgb(final_output)
            values = {
                "target": target,
                "initial": initial_rgb,
                "final": final_rgb,
                "error_x4": error,
                "alpha": alpha,
                "depth_support": depth,
            }
            artifacts = {}
            for label, value in values.items():
                path = root / f"{name}_{label}.png"
                _save_rgb(path, value)
                artifacts[label] = _artifact(path, out)
            views[name] = artifacts
            rows.append([_labeled(value, f"{name} · {label}") for label, value in values.items()])
    sheet = root / "reporting_contact_sheet.png"
    _sheet(rows).save(sheet)
    return {"contact_sheet": _artifact(sheet, out), "views": views}


def _run_arm(
    arm: str,
    inputs: ReconstructionInputs,
    views: list[Any],
    input_record: dict[str, Any],
    features: Any,
    scene: Any,
    renderer: Any,
    out: Path,
) -> dict[str, Any]:
    root = out / "arms" / arm
    root.mkdir(parents=True, exist_ok=True)
    carve = _carve_config()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    lift_started = time.perf_counter()
    if arm == "interior":
        initialization = CompactCarveInitializer(carve).initialize(
            inputs,
            backends=[view.query_backend for view in views],
            progress_callback=make_placement_progress_printer(every_batches=20, every_seconds=20),
        )
        cover_started = time.perf_counter()
        covered = reconcile_covariances(
            initialization.gaussians,
            SurfelInitConfig(use_resolution_floor=False),
        )
        cover_seconds = time.perf_counter() - cover_started
        lift_diagnostics = {
            **initialization.diagnostics,
            "surface_cover_seconds": cover_seconds,
            "surface_cover": covered.diagnostics,
        }
        initialization = dataclasses.replace(
            initialization,
            gaussians=covered.gaussians,
            diagnostics=lift_diagnostics,
        )
    else:
        posterior = initialize_occlusion_aware_ray_posterior(
            inputs,
            views,
            carve,
            features,
            _posterior_config(arm == "posterior_reciprocal"),
            progress_callback=lambda completed, total: (
                print(f"  [posterior] {arm} {completed}/{total}", flush=True)
                if completed == total or completed % 5_000 == 0
                else None
            ),
        )
        initialization = posterior.initialization
        lift_diagnostics = posterior.diagnostics
    torch.cuda.synchronize()
    lift_seconds = time.perf_counter() - lift_started
    initial = initialization.gaussians.to("cpu")
    init_npz = root / "gaussians_init.npz"
    init_ply = root / "gaussians_init.ply"
    initial.save_npz(init_npz)
    initial.save_ply(init_ply)
    curves = [
        {
            "step": 0,
            "optimization_elapsed_seconds": 0.0,
            "n_gaussians": initial.n,
            "metrics": _evaluate_model(scene, initial.to("cuda"), renderer),
        }
    ]

    def checkpoint(snapshot: Any, step: int) -> None:
        print(f"  [eval] {arm} step={step} N={snapshot.n}", flush=True)
        curves.append(
            {
                "step": step,
                "optimization_elapsed_seconds": None,
                "n_gaussians": snapshot.n,
                "metrics": _evaluate_model(scene, snapshot, renderer),
            }
        )

    train_config = _train_config()
    print(f"[train] {arm} N={initial.n} steps={ITERATIONS}", flush=True)
    train_started = time.perf_counter()
    final, history = Trainer(train_config).train(
        scene,
        initialization.gaussians,
        checkpoint_callback=checkpoint,
    )
    torch.cuda.synchronize()
    train_wall_seconds = time.perf_counter() - train_started
    elapsed = {int(step): float(value) for step, value in history["elapsed"]}
    for record in curves:
        if record["step"] in elapsed:
            record["optimization_elapsed_seconds"] = elapsed[record["step"]]
    final_cpu = final.to("cpu")
    final_npz = root / "gaussians_final.npz"
    final_ply = root / "gaussians_final.ply"
    final_cpu.save_npz(final_npz)
    final_cpu.save_ply(final_ply)
    history_path = root / "training_history.json"
    curves_path = root / "checkpoint_metrics.json"
    _write_json(history_path, history)
    _write_json(curves_path, curves)
    visuals = _save_visuals(arm, scene, initial, final_cpu, renderer, out)
    query = []
    for name, view in zip(inputs.view_names, views, strict=True):
        backend = view.query_backend
        query.append(
            {
                "view_name": name,
                "index_entries": backend.n_entries,
                "index_payload_bytes": backend.payload_bytes,
                "pairs_evaluated": backend.total_pairs_evaluated,
                "peak_pair_chunk": backend.peak_pair_chunk,
            }
        )
    return {
        "arm": arm,
        "label": ARM_LABELS[arm],
        "status": "ok",
        "input": input_record,
        "carve_config": dataclasses.asdict(carve),
        "posterior_config": (
            None
            if arm == "interior"
            else dataclasses.asdict(_posterior_config(arm == "posterior_reciprocal"))
        ),
        "train_config": dataclasses.asdict(train_config),
        "lift_seconds": lift_seconds,
        "lift_diagnostics": lift_diagnostics,
        "query_backends": query,
        "initial_n_gaussians": initial.n,
        "final_n_gaussians": final_cpu.n,
        "training_wall_seconds_including_metrics": train_wall_seconds,
        "training_native_seconds": float(history["elapsed"][-1][1]),
        "peak_vram_gb": float(history["peak_vram_gb"]),
        "initial_metrics": curves[0]["metrics"],
        "final_metrics": curves[-1]["metrics"],
        "curve_rows": curves,
        "curves": _artifact(curves_path, out),
        "history": _artifact(history_path, out),
        "models": {
            "initial_npz": _artifact(init_npz, out),
            "initial_ply": _artifact(init_ply, out),
            "final_npz": _artifact(final_npz, out),
            "final_ply": _artifact(final_ply, out),
        },
        "visuals": visuals,
    }


def _terminal_row(record: dict[str, Any]) -> dict[str, Any]:
    if record["status"] != "ok":
        return {"arm": record["arm"], "status": record["status"], "error": record["error"]}
    metrics = record["final_metrics"]["reporting"]["aggregate"]
    initial = record["initial_metrics"]["reporting"]["aggregate"]
    packet_bytes = record["input"]["complete_packet_bytes"]
    source_bytes = record["input"]["original_source_bytes"]
    model_bytes = record["models"]["final_npz"]["bytes"]
    return {
        "arm": record["arm"],
        "status": "ok",
        "input_bytes": packet_bytes,
        "original_source_bytes": source_bytes,
        "final_model_bytes": model_bytes,
        "original_over_packets": source_bytes / packet_bytes,
        "original_over_packets_plus_model": source_bytes / (packet_bytes + model_bytes),
        "initial_n_gaussians": record["initial_n_gaussians"],
        "final_n_gaussians": record["final_n_gaussians"],
        "initial_reporting_psnr": initial["psnr"],
        "reporting_psnr": metrics["psnr"],
        "reporting_ssim": metrics["ssim"],
        "reporting_ms_ssim": metrics["ms_ssim"],
        "reporting_lpips": metrics.get("lpips"),
        "reporting_gradient_mae": metrics["gradient_mae"],
        "reporting_p99_abs": metrics["p99_abs"],
        "lift_seconds": record["lift_seconds"],
        "feature_seconds": (
            0.0
            if record["arm"] == "interior"
            else float(record["lift_diagnostics"]["feature_seconds_shared"])
        ),
        "pretraining_seconds": (
            record["input"]["cold_decode_seconds"]
            + record["input"]["adapter_seconds"]
            + record["lift_seconds"]
            + (
                0.0
                if record["arm"] == "interior"
                else float(record["lift_diagnostics"]["feature_seconds_shared"])
            )
        ),
        "training_native_seconds": record["training_native_seconds"],
        "peak_vram_gb": record["peak_vram_gb"],
    }


def _first_target(record: dict[str, Any], target: float) -> tuple[int | None, float | None]:
    for curve in record.get("curve_rows", []):
        value = curve["metrics"]["reporting"]["aggregate"]["psnr"]
        if value >= target:
            return int(curve["step"]), curve["optimization_elapsed_seconds"]
    return None, None


def _decision(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(row["status"] != "ok" for row in rows):
        return {
            "advance": False,
            "scalar_pass": False,
            "manual_visual_review_required": True,
            "reason": "one or more arms failed",
        }
    by_arm = {row["arm"]: row for row in rows}
    complete = by_arm["posterior_reciprocal"]
    controls = [by_arm["interior"], by_arm["posterior_no_reciprocal"]]
    strongest = max(controls, key=lambda row: row["reporting_psnr"])
    record_by_arm = {record["arm"]: record for record in records}
    control_step, control_seconds = _first_target(
        record_by_arm[strongest["arm"]], strongest["reporting_psnr"]
    )
    candidate_step, candidate_seconds = _first_target(
        record_by_arm["posterior_reciprocal"], strongest["reporting_psnr"]
    )
    gates = {
        "terminal_reporting_psnr_within_0_1db": (
            complete["reporting_psnr"] >= strongest["reporting_psnr"] - 0.1
        ),
        "terminal_reporting_ms_ssim_within_0_01": (
            complete["reporting_ms_ssim"] >= strongest["reporting_ms_ssim"] - 0.01
        ),
        "terminal_reporting_lpips_no_worse": (
            complete["reporting_lpips"] is not None
            and strongest["reporting_lpips"] is not None
            and complete["reporting_lpips"] <= strongest["reporting_lpips"]
        ),
        "terminal_reporting_gradient_mae_no_worse": (
            complete["reporting_gradient_mae"] <= strongest["reporting_gradient_mae"]
        ),
        "control_terminal_psnr_reached_no_later": (
            candidate_step is not None
            and control_step is not None
            and candidate_step <= control_step
        ),
        "pretraining_wall_at_most_2x_interior": (
            complete["pretraining_seconds"] <= 2.0 * by_arm["interior"]["pretraining_seconds"]
        ),
        "complete_scene_representation_smaller_than_original_jpegs": (
            complete["original_over_packets_plus_model"] > 1.0
        ),
        "all_initial_counts_exact": all(
            row["initial_n_gaussians"] == INITIAL_GAUSSIANS for row in rows
        ),
        "all_final_counts_within_cap": all(
            row["final_n_gaussians"] <= MAX_GAUSSIANS for row in rows
        ),
        "all_arms_reuse_identical_packet_bytes": len({row["input_bytes"] for row in rows}) == 1,
    }
    return {
        "advance": False,
        "scalar_pass": all(gates.values()),
        "manual_visual_review_required": True,
        "artifact_gate": "no directional doubles, floaters, or conspicuous geometry trails",
        "strongest_control": strongest["arm"],
        "gates": gates,
        "candidate_minus_control": {
            "reporting_psnr_db": complete["reporting_psnr"] - strongest["reporting_psnr"],
            "reporting_ms_ssim": complete["reporting_ms_ssim"] - strongest["reporting_ms_ssim"],
            "reporting_lpips": (
                None
                if complete["reporting_lpips"] is None or strongest["reporting_lpips"] is None
                else complete["reporting_lpips"] - strongest["reporting_lpips"]
            ),
            "reporting_gradient_mae": (
                complete["reporting_gradient_mae"] - strongest["reporting_gradient_mae"]
            ),
        },
        "convergence_to_control_terminal": {
            "control_step": control_step,
            "control_seconds": control_seconds,
            "candidate_step": candidate_step,
            "candidate_seconds": candidate_seconds,
        },
    }


def _write_metrics(out: Path, rows: list[dict[str, Any]]) -> None:
    _write_json(out / "metrics.json", rows)
    with (out / "metrics.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    keys = sorted({key for row in rows for key in row})
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(records: list[dict[str, Any]], out: Path) -> Path:
    metrics = (
        ("psnr", "PSNR (dB)"),
        ("ms_ssim", "MS-SSIM"),
        ("lpips", "LPIPS"),
        ("gradient_mae", "Gradient MAE"),
        ("p99_abs", "P99 absolute error"),
        ("alpha_mean", "Mean alpha"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
        for record in records:
            if record["status"] != "ok":
                continue
            points = []
            for curve in record["curve_rows"]:
                value = curve["metrics"]["reporting"]["aggregate"].get(metric)
                if value is not None:
                    points.append((curve["step"], value))
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    markersize=3,
                    label=record["arm"],
                )
        axis.axvline(FIXED_PREFIX_STEPS, color="black", linestyle="--", alpha=0.4)
        axis.set_xlabel("attempted optimization step")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    figure.tight_layout()
    path = out / "all_metric_curves.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_html(
    out: Path,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    decision: dict[str, Any],
    curves: Path,
) -> None:
    table_rows = []
    for row in rows:
        if row["status"] != "ok":
            table_rows.append(
                f"<tr><td>{escape(row['arm'])}</td><td colspan='12'>{escape(row['error'])}</td></tr>"
            )
            continue
        table_rows.append(
            "<tr>"
            f"<td>{escape(row['arm'])}</td>"
            f"<td>{row['final_n_gaussians']:,}</td>"
            f"<td>{_fmt(row['reporting_psnr'])}</td>"
            f"<td>{_fmt(row['reporting_ms_ssim'])}</td>"
            f"<td>{_fmt(row['reporting_lpips'])}</td>"
            f"<td>{_fmt(row['reporting_gradient_mae'])}</td>"
            f"<td>{_fmt(row['pretraining_seconds'], 2)}</td>"
            f"<td>{_fmt(row['training_native_seconds'], 2)}</td>"
            f"<td>{row['input_bytes']:,}</td>"
            f"<td>{row['final_model_bytes']:,}</td>"
            f"<td>{_fmt(row['original_over_packets'], 2)}×</td>"
            f"<td>{_fmt(row['original_over_packets_plus_model'], 2)}×</td>"
            "</tr>"
        )
    cards = []
    for record in records:
        if record["status"] != "ok":
            continue
        sheet = record["visuals"]["contact_sheet"]["path"]
        cards.append(
            f"<section><h2>{escape(record['label'])}</h2>"
            f"<p><a href='{escape(sheet)}'><img src='{escape(sheet)}' loading='lazy'></a></p>"
            f"<p><a href='{escape(record['curves']['path'])}'>checkpoint metrics</a> · "
            f"<a href='{escape(record['history']['path'])}'>training history</a> · "
            f"<a href='{escape(record['models']['final_npz']['path'])}'>final model</a></p></section>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>CORE-018 ray-posterior diagnostic</title>
<style>body{{font-family:sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.4rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;background:#f4f4f4;padding:1rem}}</style></head>
<body><h1>CORE-018 occlusion-aware ray-posterior diagnostic</h1>
<p><strong>Scope:</strong> one disjoint unmasked karate scene, one seed, reduced resolution,
development only. Scalar success cannot override native visual failure.</p>
<p><a href="manifest.json">manifest</a> · <a href="plan.json">plan</a> ·
<a href="metrics.json">JSON</a> · <a href="metrics.jsonl">JSONL</a> ·
<a href="metrics.csv">CSV</a></p>
<table><thead><tr><th>Arm</th><th>Final N</th><th>PSNR</th><th>MS-SSIM</th><th>LPIPS</th><th>Grad MAE</th><th>Pretrain s</th><th>Train s</th><th>Packet B</th><th>Model B</th><th>Original/packet</th><th>Original/(packet+model)</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>All metric curves</h2><p><a href="{escape(curves.name)}"><img src="{escape(curves.name)}"></a></p>
<h2>Fail-closed decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
{''.join(cards)}</body></html>"""
    (out / "index.html").write_text(html)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--realtime-gs-root", type=Path, default=DEFAULT_RTGS_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = args.out.expanduser().resolve()
    frame = args.frame.expanduser().resolve()
    rtgs_root = args.realtime_gs_root.expanduser().resolve()
    calibration = frame.parent / "calibration_dome.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")
    if not frame.is_dir() or not calibration.is_file():
        raise FileNotFoundError("karate frame or calibration is missing")
    if not torch.cuda.is_available():
        raise RuntimeError("CORE-018 diagnostic requires CUDA")
    out.mkdir(parents=True)
    started = time.perf_counter()
    repositories = {
        "structsplat": _repository_record(STRUCTSPLAT_ROOT),
        "realtime_gs": _repository_record(rtgs_root),
    }
    snapshots = _snapshot_sources(out, rtgs_root)
    packet_scene = load_calibrated_scene(
        frame,
        calibration_path=calibration,
        downscale=PACKET_DOWNSCALE,
        view_ids=TRAIN_CAMERA_IDS,
        test_every=0,
        load_masks=False,
        undistort=True,
    )
    scene = load_calibrated_scene(
        frame,
        calibration_path=calibration,
        downscale=EVALUATION_DOWNSCALE,
        view_ids=TRAIN_CAMERA_IDS + REPORT_CAMERA_IDS,
        test_every=0,
        load_masks=False,
        undistort=True,
    )
    if packet_scene.bounds_hint is None or scene.bounds_hint is None:
        raise RuntimeError("the calibrated loader must provide explicit shared bounds")
    scene.train_indices = list(range(len(TRAIN_CAMERA_IDS)))
    scene.test_indices = list(range(len(TRAIN_CAMERA_IDS), len(TRAIN_CAMERA_IDS + REPORT_CAMERA_IDS)))
    scene.name = "core018-karate-frame00060"
    scene.validate()
    plan = {
        "schema": "core018.ray_posterior.plan.v1",
        "scope": "single_disjoint_scene_single_seed_reduced_resolution_diagnostic",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "repositories": repositories,
        "environment": _environment(),
        "source_snapshots": snapshots,
        "frame": str(frame),
        "calibration": {
            "path": str(calibration),
            "bytes": calibration.stat().st_size,
            "sha256": _sha256(calibration),
        },
        "train_camera_ids": list(TRAIN_CAMERA_IDS),
        "report_camera_ids": list(REPORT_CAMERA_IDS),
        "train_view_names": list(packet_scene.view_names),
        "report_view_names": list(scene.view_names[len(TRAIN_CAMERA_IDS) :]),
        "packet_downscale": PACKET_DOWNSCALE,
        "evaluation_downscale": EVALUATION_DOWNSCALE,
        "bounds_hint": {
            "center": packet_scene.bounds_hint[0].tolist(),
            "extent": packet_scene.bounds_hint[1],
        },
        "arms": list(ARMS),
        "carve_config": dataclasses.asdict(_carve_config()),
        "posterior_no_reciprocal": dataclasses.asdict(_posterior_config(False)),
        "posterior_reciprocal": dataclasses.asdict(_posterior_config(True)),
        "train_config": dataclasses.asdict(_train_config()),
        "fixed_topology_prefix_steps": FIXED_PREFIX_STEPS,
        "decision_gate": {
            "psnr_delta_min_db": -0.1,
            "ms_ssim_delta_min": -0.01,
            "lpips_no_worse": True,
            "gradient_mae_no_worse": True,
            "time_to_control_terminal_no_later": True,
            "pretraining_over_interior_max": 2.0,
            "original_over_packets_plus_model_min": 1.0,
            "manual_native_visual_review_required": True,
        },
    }
    _write_json(out / "plan.json", plan)
    print("[packets] build shared construction packets", flush=True)
    packet_record = _build_shared_packets(packet_scene, frame, out)
    _write_json(out / "shared_packets.json", packet_record)

    print("[features] build shared packet-derived pyramids", flush=True)
    feature_inputs, feature_views, _ = _reload_packets(packet_record, packet_scene, out)
    features = build_packet_feature_pyramids(
        feature_views,
        _posterior_config(True),
        progress_callback=lambda completed, total: print(
            f"  [features] {completed}/{total}", flush=True
        ),
    )
    _write_json(out / "feature_receipt.json", features.diagnostics)
    del feature_inputs
    gc.collect()
    torch.cuda.empty_cache()

    renderer = get_rasterizer(
        "gsplat", device=torch.device("cuda"), packed=False, antialiased=True
    )
    records = []
    for arm in ARMS:
        inputs = views = None
        try:
            print(f"[arm] {arm}: cold reload", flush=True)
            inputs, views, input_record = _reload_packets(packet_record, packet_scene, out)
            record = _run_arm(
                arm,
                inputs,
                views,
                input_record,
                features,
                scene,
                renderer,
                out,
            )
        except Exception as error:
            record = {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            print(record["traceback"], file=sys.stderr, flush=True)
        records.append(record)
        _write_json(out / "partial_records.json", records)
        del inputs, views
        gc.collect()
        torch.cuda.empty_cache()
    rows = [_terminal_row(record) for record in records]
    _write_metrics(out, rows)
    decision = _decision(records, rows)
    _write_json(out / "decision.json", decision)
    curves = _plot_curves(records, out)
    _write_html(out, rows, records, decision, curves)
    manifest_records = []
    for record in records:
        retained = dict(record)
        retained.pop("curve_rows", None)
        manifest_records.append(retained)
    manifest = {
        "schema": "core018.ray_posterior.manifest.v1",
        "status": "ok" if all(record["status"] == "ok" for record in records) else "partial",
        "scope": plan["scope"],
        "plan": _artifact(out / "plan.json", out),
        "shared_packets": _artifact(out / "shared_packets.json", out),
        "feature_receipt": _artifact(out / "feature_receipt.json", out),
        "records": manifest_records,
        "decision": decision,
        "metrics": {
            name: _artifact(out / name, out)
            for name in ("metrics.json", "metrics.jsonl", "metrics.csv")
        },
        "plots": {"all_metric_curves": _artifact(curves, out)},
        "report": _artifact(out / "index.html", out),
        "total_wall_seconds": time.perf_counter() - started,
    }
    _write_json(out / "manifest.json", manifest)
    print(json.dumps({"decision": decision, "rows": rows}, indent=2), flush=True)
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
