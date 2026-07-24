#!/usr/bin/env python3
"""Compare FIT-021/FIT-022 pooled boundary variants on Janelle frame_00008/C0001.

The experiment deliberately uses one frozen, boundary-sampled 5k initialization for every arm.
Each arm gets the same 11k pool capacity, optimizer, topology budgets, schedule, and fixed-
topology terminal settle.  The only arm-level difference is the FIT-022 coverage target:

* pooled triage alone (FIT-021 control),
* tensor coverage matching,
* tensor + boundary-band coverage matching,
* tensor + current-error-blend coverage matching.

Besides native reconstructions and fixed-scale error images, the runner instruments the pooled
site activator without changing its implementation.  For every event it records the selected
pixel coordinates, their residual rank, raw coverage, initialized covariance/opacity, and the
error immediately before activation, immediately after the complete triage event, before the
next event, and at the terminal field.  This makes "births go to high error and reduce it" a
measured property rather than an inference from global PSNR.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import csv
from dataclasses import asdict, dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for source_root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    source_text = str(source_root.resolve())
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)

from fit_janelle_complete_refinement import (  # noqa: E402
    DEFAULT_CAPTURE_ROOT,
    DEFAULT_REALTIME_ROOT,
    _atomic_json,
    _atomic_text,
    _error_heatmap,
    _find_source,
    _git_record,
    _initial_field,
    _load_bridge,
    _render_field,
    _resize_hwc,
    _to_image,
)


DEFAULT_OUT = REPOSITORY_ROOT / "runs/janelle_C0001_pooled_boundary_variants_20260723"
DEFAULT_HISTORICAL = REPOSITORY_ROOT / "runs/janelle_C0001_complete_refinement_20260722"
HOLE_COVERAGE_TAU = 0.05


@dataclass(frozen=True)
class ArmSpec:
    key: str
    label: str
    coverage_target: str | None
    description: str


ARMS = (
    ArmSpec(
        "pooled_triage",
        "Pooled triage",
        None,
        "FIT-021 park → merge → split → high-error spawn; no coverage regularizer.",
    ),
    ArmSpec(
        "coverage_tensor",
        "Triage + tensor",
        "tensor",
        "FIT-021 plus feature-weighted two-sided coverage matching.",
    ),
    ArmSpec(
        "coverage_tensor_boundary",
        "Triage + tensor boundary",
        "tensor_boundary",
        "FIT-021 plus explicit extra coverage target in the inside mask-boundary band.",
    ),
    ArmSpec(
        "coverage_error_blend",
        "Triage + error blend",
        "error_blend",
        "FIT-021 plus a detached blend of tensor structure and current pixel residual.",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_snapshots(out: Path) -> list[dict[str, Any]]:
    sources = (
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "deprecated_scripts/fit_janelle_complete_refinement.py",
        REPOSITORY_ROOT / "deprecated_scripts/fit_janelle_mask_contained.py",
        REPOSITORY_ROOT / "src/structsplat/config.py",
        REPOSITORY_ROOT / "src/structsplat/fit.py",
        REPOSITORY_ROOT / "src/structsplat/pool.py",
        REPOSITORY_ROOT / "src/structsplat/triage.py",
    )
    records: list[dict[str, Any]] = []
    for source in sources:
        payload = source.read_bytes()
        relative = Path("provenance") / source.relative_to(REPOSITORY_ROOT)
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        records.append(
            {
                "source": str(source),
                "snapshot": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    return records


def _psnr(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-12))


def _quantile(values, q: float) -> float:
    import torch

    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.float(), q))


def _live_count(field) -> int:
    from structsplat.pool import POOL_PARK_COORD

    return int((field.means.detach()[:, 0] > POOL_PARK_COORD * 0.5).sum())


def _coverage_visual(denominator, mask):
    import torch

    normalized = (1.0 - torch.exp(-denominator.clamp_min(0.0))).clamp(0.0, 1.0)
    rgb = torch.stack(
        [0.12 * normalized, 0.55 * normalized, normalized], dim=-1
    )
    return rgb * mask[..., None].to(rgb)


def _hole_visual(target, denominator, mask, boundary, tau: float):
    import torch

    base = target.clamp(0.0, 1.0) * 0.28
    holes = mask & (denominator < tau)
    out = base.clone()
    out[holes] = torch.tensor([1.0, 0.05, 0.25], device=out.device, dtype=out.dtype)
    # Cyan identifies the measured 5x5 inside boundary band; red still wins at actual holes.
    out[boundary & ~holes] = (
        0.45 * out[boundary & ~holes]
        + 0.55 * torch.tensor([0.0, 0.9, 1.0], device=out.device, dtype=out.dtype)
    )
    return out


def _metric_record(render, target, mask, boundary, denominator, tau: float) -> dict[str, Any]:
    import torch

    clamped = render.clamp(0.0, 1.0)
    pixel_abs = (clamped - target).abs().mean(dim=2)
    pixel_sq = (render - target).square().mean(dim=2)
    fg_abs = pixel_abs[mask]
    fg_sq = pixel_sq[mask]
    boundary_abs = pixel_abs[boundary]
    boundary_sq = pixel_sq[boundary]
    interior = mask & ~boundary
    interior_abs = pixel_abs[interior]
    outside = render[~mask].abs()
    holes = mask & (denominator < tau)
    error_holes = holes & (pixel_abs > 0.05)
    top_count = max(1, int(math.ceil(0.01 * max(int(fg_abs.numel()), 1))))
    top1 = torch.topk(fg_abs, k=min(top_count, int(fg_abs.numel()))).values
    return {
        "foreground_psnr_raw_db": _psnr(float(fg_sq.mean())),
        "foreground_mae_clamped": float(fg_abs.mean()),
        "foreground_error_p50": _quantile(fg_abs, 0.50),
        "foreground_error_p90": _quantile(fg_abs, 0.90),
        "foreground_error_p95": _quantile(fg_abs, 0.95),
        "foreground_error_p99": _quantile(fg_abs, 0.99),
        "foreground_error_top1pct_mean": float(top1.mean()),
        "foreground_error_max": float(fg_abs.max()),
        "foreground_fraction_error_gt_005": float((fg_abs > 0.05).float().mean()),
        "foreground_fraction_error_gt_010": float((fg_abs > 0.10).float().mean()),
        "boundary_psnr_raw_db": _psnr(float(boundary_sq.mean())),
        "boundary_mae_clamped": float(boundary_abs.mean()),
        "boundary_error_p95": _quantile(boundary_abs, 0.95),
        "boundary_error_p99": _quantile(boundary_abs, 0.99),
        "interior_mae_clamped": float(interior_abs.mean()),
        "coverage_tau": float(tau),
        "coverage_mean_inside": float(denominator[mask].mean()),
        "coverage_p01_inside": _quantile(denominator[mask], 0.01),
        "undercoverage_pixels": int(holes.sum()),
        "undercoverage_fraction": float(holes.sum() / mask.sum().clamp_min(1)),
        "boundary_undercoverage_fraction": float(
            (holes & boundary).sum() / boundary.sum().clamp_min(1)
        ),
        "high_error_undercoverage_pixels": int(error_holes.sum()),
        "high_error_undercoverage_fraction": float(
            error_holes.sum() / mask.sum().clamp_min(1)
        ),
        "outside_mask_max_abs": float(outside.max()) if outside.numel() else 0.0,
        "outside_mask_mean_abs": float(outside.mean()) if outside.numel() else 0.0,
        "outside_mask_exact_zero": bool(not outside.numel() or float(outside.max()) == 0.0),
        "render_out_of_range_fraction": float(
            ((render < 0.0) | (render > 1.0)).float().mean()
        ),
        "finite": bool(torch.isfinite(render).all() and torch.isfinite(denominator).all()),
    }


def _evaluate(field, cfg, target, mask, boundary, tau: float):
    import torch
    from structsplat.fit import _raw_weight_map_field

    height, width = mask.shape
    with torch.no_grad():
        render = _render_field(field, cfg, height, width)
        denominator = _raw_weight_map_field(
            field,
            cfg,
            height,
            width,
            support_fade_alpha=1.0,
            detach_opacities=False,
        ).view(height, width)
        metrics = _metric_record(
            render, target, mask, boundary, denominator, tau
        )
    return render, denominator, metrics


def _save_preview(path: Path, value, width: int, *, quality: int = 93) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_image(_resize_hwc(value, width)).save(path, quality=quality, subsampling=0)


class ProgressWriter:
    def __init__(self, out: Path, label: str, target, mask, boundary, preview_width: int,
                 tau: float):
        self.out = out
        self.label = label
        self.target = target
        self.mask = mask
        self.boundary = boundary
        self.preview_width = int(preview_width)
        self.tau = float(tau)
        self.records: list[dict[str, Any]] = []
        self.started = time.perf_counter()

    def snapshot(self, field, cfg, *, stage: str, iteration: int, event: str,
                 loss: float | None = None, native: bool = False) -> dict[str, Any]:
        render, denominator, metrics = _evaluate(
            field, cfg, self.target, self.mask, self.boundary, self.tau
        )
        mask3 = self.mask[..., None].to(render)
        error = (render.clamp(0.0, 1.0) - self.target).abs().mean(dim=2) * self.mask
        ordinal = len(self.records)
        stem = f"{ordinal:03d}_iter_{int(iteration):06d}_{stage}_{event}"
        recon_rel = Path("images/intermediates") / f"{stem}_reconstruction.jpg"
        error_rel = Path("images/intermediates") / f"{stem}_absolute_error_x4.jpg"
        hole_rel = Path("images/intermediates") / f"{stem}_undercoverage.jpg"
        _save_preview(self.out / recon_rel, render.clamp(0.0, 1.0) * mask3, self.preview_width)
        _save_preview(
            self.out / error_rel, _error_heatmap(error), self.preview_width, quality=95
        )
        _save_preview(
            self.out / hole_rel,
            _hole_visual(self.target, denominator, self.mask, self.boundary, self.tau),
            self.preview_width,
            quality=95,
        )
        native_paths = None
        if native:
            native_paths = {
                "reconstruction": "images/final_reconstruction.png",
                "absolute_error_x4": "images/final_absolute_error_x4.png",
                "coverage": "images/final_coverage.png",
                "undercoverage": "images/final_undercoverage.png",
            }
            _to_image(render.clamp(0.0, 1.0) * mask3).save(
                self.out / native_paths["reconstruction"]
            )
            _to_image(_error_heatmap(error)).save(
                self.out / native_paths["absolute_error_x4"]
            )
            _to_image(_coverage_visual(denominator, self.mask)).save(
                self.out / native_paths["coverage"]
            )
            _to_image(
                _hole_visual(self.target, denominator, self.mask, self.boundary, self.tau)
            ).save(self.out / native_paths["undercoverage"])
        record = {
            "ordinal": ordinal,
            "stage": stage,
            "event": event,
            "iteration": int(iteration),
            "n_rows_tensor": int(field.n),
            "n_live": _live_count(field),
            "loss": None if loss is None else float(loss),
            "elapsed_seconds": time.perf_counter() - self.started,
            **metrics,
            "reconstruction": recon_rel.as_posix(),
            "absolute_error_x4": error_rel.as_posix(),
            "undercoverage": hole_rel.as_posix(),
            "native": native_paths,
        }
        self.records.append(record)
        self.flush()
        print(
            f"[{self.label}] snapshot {ordinal:03d} iter={iteration:5d} "
            f"live={record['n_live']:5d} fg={metrics['foreground_psnr_raw_db']:.3f} dB "
            f"boundary={metrics['boundary_psnr_raw_db']:.3f} dB "
            f"p99={metrics['foreground_error_p99']:.4f} "
            f"holes={100.0 * metrics['undercoverage_fraction']:.3f}%",
            flush=True,
        )
        return record

    def flush(self) -> None:
        _atomic_json(self.out / "progress.json", {"snapshots": self.records})
        if not self.records:
            return
        fields = list(self.records[0])
        temporary = self.out / f".progress.csv.tmp-{os.getpid()}"
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for record in self.records:
                writer.writerow({key: record.get(key) for key in fields})
        os.replace(temporary, self.out / "progress.csv")


class SpawnDiagnostics(AbstractContextManager):
    """Temporary experiment-only wrapper around triage._activate_sites."""

    def __init__(self, out: Path, cfg, target, mask, boundary, preview_width: int):
        import structsplat.triage as triage

        self.out = out
        self.cfg = cfg
        self.target = target
        self.mask = mask
        self.boundary = boundary
        self.preview_width = int(preview_width)
        self.height, self.width = mask.shape
        self.module = triage
        self.original = triage._activate_sites
        self.records: list[dict[str, Any]] = []

    def __enter__(self):
        self.module._activate_sites = self._wrapped
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.module._activate_sites = self.original
        return False

    def _write(self) -> None:
        _atomic_json(self.out / "spawn_events.json", {"events": self.records})

    def _update_previous_before_next(self, render_img) -> None:
        if not self.records:
            return
        previous = self.records[-1]
        if previous.get("error_before_next_event") is not None:
            return
        sites = previous.get("site_indices", [])
        if not sites:
            previous["error_before_next_event"] = []
            previous["error_before_next_event_mean"] = None
            return
        import torch

        indices = torch.as_tensor(sites, device=render_img.device, dtype=torch.long)
        error = (render_img.detach() - self.target).abs().mean(dim=2).reshape(-1)[indices]
        previous["error_before_next_event"] = error.detach().cpu().tolist()
        previous["error_before_next_event_mean"] = float(error.mean())

    def _draw_sites(self, error, xy, is_boundary, event_number: int) -> str:
        preview = _to_image(
            _resize_hwc(_error_heatmap(error * self.mask), self.preview_width)
        )
        draw = ImageDraw.Draw(preview)
        scale = preview.width / self.width
        radius = max(1.5, 2.0 * scale)
        xy_cpu = xy.detach().cpu().tolist()
        boundary_cpu = is_boundary.detach().cpu().tolist()
        for (x, y), on_boundary in zip(xy_cpu, boundary_cpu):
            px, py = float(x) * scale, float(y) * scale
            color = (0, 255, 255) if on_boundary else (255, 255, 0)
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                outline=color,
                width=1,
            )
        relative = Path("images/spawn_events") / f"event_{event_number:02d}_selected_sites.jpg"
        destination = self.out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        preview.save(destination, quality=95, subsampling=0)
        return relative.as_posix()

    def _wrapped(self, field, pool, stats, cfg, target, render_img, mask_constraint,
                 free, k_budget):
        import torch

        self._update_previous_before_next(render_img)
        # Render the already parked/merged/split field before activation.  Comparing against the
        # post-event observer isolates the spawn contribution much more closely than comparing
        # with the stale pre-optimizer render used for site ranking.
        with torch.no_grad():
            pre_activation_render = _render_field(
                field, cfg, self.height, self.width
            )
        rows, count, holes = self.original(
            field,
            pool,
            stats,
            cfg,
            target,
            render_img,
            mask_constraint,
            free,
            k_budget,
        )
        event_number = len(self.records) + 1
        record: dict[str, Any] = {
            "event": event_number,
            "iteration": event_number * int(cfg.triage_every),
            "requested": int(k_budget),
            "spawned": int(count),
            "spawned_holes_core": int(holes),
            "post_event_error": None,
            "error_before_next_event": None,
            "terminal_error": None,
        }
        if count <= 0:
            record.update(
                site_indices=[],
                site_xy=[],
                pre_score_error=[],
                pre_activation_error=[],
            )
            self.records.append(record)
            self._write()
            return rows, count, holes

        means = field.means.detach()[rows]
        x = torch.round(means[:, 0]).long().clamp(0, self.width - 1)
        y = torch.round(means[:, 1]).long().clamp(0, self.height - 1)
        sites = y * self.width + x
        residual = (render_img.detach() - target.detach()).abs().mean(dim=2)
        pre_score = residual.reshape(-1)[sites]
        pre_activation = (
            (pre_activation_render - target.detach()).abs().mean(dim=2).reshape(-1)[sites]
        )
        selected_coverage = stats["denominator"].reshape(-1)[sites]
        allowed = (
            mask_constraint.eroded_flat
            if mask_constraint is not None and cfg.mask_contain
            else self.mask.reshape(-1)
        )
        allowed_errors = residual.reshape(-1)[allowed]
        quantiles = torch.quantile(
            allowed_errors.float(), torch.tensor([0.95, 0.99], device=allowed_errors.device)
        )
        is_boundary = self.boundary.reshape(-1)[sites]
        scales = field.scales().detach()[rows]
        major = scales.max(dim=1).values
        minor = scales.min(dim=1).values.clamp_min(1e-8)
        opacities = field.opacity_values().detach()[rows]
        overlay = self._draw_sites(residual, torch.stack([x, y], dim=1), is_boundary, event_number)
        record.update(
            {
                "site_indices": sites.detach().cpu().tolist(),
                "site_xy": torch.stack([x, y], dim=1).detach().cpu().tolist(),
                "pre_score_error": pre_score.detach().cpu().tolist(),
                "pre_score_error_mean": float(pre_score.mean()),
                "pre_score_error_min": float(pre_score.min()),
                "pre_score_error_max": float(pre_score.max()),
                "pre_activation_error": pre_activation.detach().cpu().tolist(),
                "pre_activation_error_mean": float(pre_activation.mean()),
                "selected_coverage": selected_coverage.detach().cpu().tolist(),
                "selected_coverage_mean": float(selected_coverage.mean()),
                "selected_coverage_min": float(selected_coverage.min()),
                "allowed_error_p95": float(quantiles[0]),
                "allowed_error_p99": float(quantiles[1]),
                "selected_above_allowed_p99_fraction": float(
                    (pre_score >= quantiles[1]).float().mean()
                ),
                "boundary_site_count": int(is_boundary.sum()),
                "boundary_site_fraction": float(is_boundary.float().mean()),
                "initialized_major_scale_mean_px": float(major.mean()),
                "initialized_minor_scale_mean_px": float(minor.mean()),
                "initialized_axis_ratio_mean": float((major / minor).mean()),
                "initialized_opacity_mean": float(opacities.mean()),
                "selected_sites_overlay": overlay,
            }
        )
        self.records.append(record)
        self._write()
        return rows, count, holes

    def post_event(self, field, iteration: int) -> None:
        import torch

        candidates = [
            record
            for record in self.records
            if record["iteration"] == int(iteration) and record["post_event_error"] is None
        ]
        if not candidates:
            return
        record = candidates[-1]
        sites = record.get("site_indices", [])
        if not sites:
            record["post_event_error"] = []
            record["post_event_error_mean"] = None
            self._write()
            return
        with torch.no_grad():
            render = _render_field(field, self.cfg, self.height, self.width)
            indices = torch.as_tensor(sites, device=render.device, dtype=torch.long)
            error = (render - self.target).abs().mean(dim=2).reshape(-1)[indices]
        record["post_event_error"] = error.detach().cpu().tolist()
        record["post_event_error_mean"] = float(error.mean())
        before = np.asarray(record["pre_activation_error"], dtype=np.float64)
        after = np.asarray(record["post_event_error"], dtype=np.float64)
        record["post_event_improved_fraction"] = float(np.mean(after < before))
        record["post_event_mean_reduction"] = float(np.mean(before - after))
        self._write()

    def finalize(self, terminal_render) -> None:
        import torch

        terminal_flat = (
            terminal_render.detach() - self.target
        ).abs().mean(dim=2).reshape(-1)
        for record in self.records:
            sites = record.get("site_indices", [])
            if not sites:
                record["terminal_error"] = []
                record["terminal_error_mean"] = None
                continue
            indices = torch.as_tensor(
                sites, device=terminal_flat.device, dtype=torch.long
            )
            error = terminal_flat[indices]
            record["terminal_error"] = error.detach().cpu().tolist()
            record["terminal_error_mean"] = float(error.mean())
            before = np.asarray(record["pre_score_error"], dtype=np.float64)
            final = np.asarray(record["terminal_error"], dtype=np.float64)
            record["terminal_improved_fraction"] = float(np.mean(final < before))
            record["terminal_mean_reduction"] = float(np.mean(before - final))
        self._write()


def _aggregate_spawn(records: list[dict[str, Any]]) -> dict[str, Any]:
    def concatenate(key: str) -> np.ndarray:
        chunks = [
            np.asarray(record.get(key, []), dtype=np.float64)
            for record in records
            if record.get(key)
        ]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)

    pre_score = concatenate("pre_score_error")
    pre_activation = concatenate("pre_activation_error")
    post = concatenate("post_event_error")
    terminal = concatenate("terminal_error")
    coverage = concatenate("selected_coverage")
    total = int(sum(record.get("spawned", 0) for record in records))
    result: dict[str, Any] = {
        "events": len(records),
        "spawned_total": total,
        "spawned_holes_core_total": int(
            sum(record.get("spawned_holes_core", 0) for record in records)
        ),
        "boundary_sites_total": int(
            sum(record.get("boundary_site_count", 0) for record in records)
        ),
        "selected_above_allowed_p99_weighted_fraction": None,
    }
    if total:
        result["selected_above_allowed_p99_weighted_fraction"] = float(
            sum(
                record.get("selected_above_allowed_p99_fraction", 0.0)
                * record.get("spawned", 0)
                for record in records
            )
            / total
        )
    if pre_score.size:
        result.update(
            {
                "pre_score_error_mean": float(pre_score.mean()),
                "pre_score_error_p50": float(np.quantile(pre_score, 0.50)),
                "pre_score_error_p95": float(np.quantile(pre_score, 0.95)),
                "selected_coverage_mean": float(coverage.mean()),
            }
        )
    if pre_activation.size and post.size:
        result.update(
            {
                "pre_activation_error_mean": float(pre_activation.mean()),
                "post_event_error_mean": float(post.mean()),
                "post_event_mean_reduction": float((pre_activation - post).mean()),
                "post_event_relative_reduction": float(
                    (pre_activation.mean() - post.mean())
                    / max(pre_activation.mean(), 1e-12)
                ),
                "post_event_improved_fraction": float(
                    np.mean(post < pre_activation)
                ),
            }
        )
    if pre_score.size and terminal.size:
        result.update(
            {
                "terminal_error_at_spawn_sites_mean": float(terminal.mean()),
                "terminal_mean_reduction_at_spawn_sites": float(
                    (pre_score - terminal).mean()
                ),
                "terminal_relative_reduction_at_spawn_sites": float(
                    (pre_score.mean() - terminal.mean())
                    / max(pre_score.mean(), 1e-12)
                ),
                "terminal_improved_spawn_site_fraction": float(
                    np.mean(terminal < pre_score)
                ),
            }
        )
    return result


def _triage_config(args: argparse.Namespace, spec: ArmSpec):
    from structsplat.config import FitConfig

    return FitConfig(
        iters=int(args.triage_iters),
        renderer="cuda_tiled",
        render_chunk=512,
        pixel_loss="l2",
        ssim_weight=0.0,
        loss_weighting="mask",
        mask_contain=True,
        mask_margin=float(args.mask_margin),
        mask_cap_mode="anisotropic",
        mask_cap_refresh_every=100,
        support_fade=True,
        log_every=100,
        checkpoint_policy="terminal",
        lr_means=3e-2,
        lr_scales=2e-2,
        lr_rot=8e-3,
        lr_color=3e-2,
        lr_opacity=1e-2,
        lr_schedule="cosine",
        lr_cosine_start_frac=0.75,
        split_scale=0.35,
        prune_keep_min=4_500,
        relocate_init_opacity=0.05,
        triage_every=int(args.triage_every),
        triage_spawn_count=int(args.triage_spawn_count),
        triage_split_count=int(args.triage_split_count),
        triage_merge_count=int(args.triage_merge_count),
        triage_park_count=int(args.triage_park_count),
        triage_hole_coverage_tau=float(args.hole_coverage_tau),
        triage_hole_opacity=0.9,
        triage_site_nms_radius=7,
        pool_capacity=int(args.pool_capacity),
        coverage_match_weight=(
            0.0 if spec.coverage_target is None else float(args.coverage_weight)
        ),
        coverage_match_target=spec.coverage_target or "tensor",
        coverage_match_beta=1.0,
        coverage_match_boundary_boost=2.0,
        coverage_match_boundary_band=3.0,
        coverage_match_error_alpha=0.5,
        coverage_match_decay_frac=(
            0.0
            if spec.coverage_target is None
            else float(args.coverage_decay_frac)
        ),
        compute_lpips=False,
    )


def _settle_config(args: argparse.Namespace):
    from structsplat.config import FitConfig

    return FitConfig(
        iters=int(args.settle_iters),
        renderer="cuda_tiled",
        render_chunk=512,
        pixel_loss="l2",
        ssim_weight=0.0,
        loss_weighting="mask",
        mask_contain=True,
        mask_margin=float(args.mask_margin),
        mask_cap_mode="anisotropic",
        mask_cap_refresh_every=100,
        support_fade=True,
        log_every=100,
        checkpoint_policy="terminal",
        lr_means=5e-3,
        lr_scales=3e-3,
        lr_rot=1e-3,
        lr_color=3e-3,
        lr_opacity=1e-3,
        lr_schedule="cosine",
        lr_cosine_start_frac=0.5,
        compute_lpips=False,
    )


def _fit_record(output: dict[str, Any], cfg) -> dict[str, Any]:
    return {
        "fit_config": asdict(cfg),
        "iterations_run": int(output["iterations_run"]),
        "stopped_early": bool(output["stopped_early"]),
        "n_gaussians": int(output["n_gaussians"]),
        "psnr": float(output["psnr"]),
        "ssim": float(output["ssim"]),
        "ms_ssim": float(output["ms_ssim"]),
        "fit_seconds": float(output["fit_seconds"]),
        "estimated_bpp": float(output["estimated_bpp"]),
        "pool": output.get("pool"),
        "history": output["history"],
    }


def _convergence_record(progress: list[dict[str, Any]]) -> dict[str, Any]:
    settle = [record for record in progress if record["stage"] == "settle"]
    if len(settle) < 2:
        return {
            "converged": False,
            "reason": "fewer than two settle snapshots",
        }
    final = settle[-1]
    earlier = [
        record for record in settle[:-1]
        if int(record["iteration"]) < int(final["iteration"])
    ]
    if not earlier:
        return {
            "converged": False,
            "reason": "no earlier settle snapshot at a distinct iteration",
        }
    previous = earlier[-1]
    fg_gain = final["foreground_psnr_raw_db"] - previous["foreground_psnr_raw_db"]
    boundary_gain = final["boundary_psnr_raw_db"] - previous["boundary_psnr_raw_db"]
    p99_drop = previous["foreground_error_p99"] - final["foreground_error_p99"]
    return {
        "converged": bool(abs(fg_gain) < 0.01 and abs(boundary_gain) < 0.03),
        "criterion": (
            "absolute foreground PSNR change < 0.01 dB and absolute boundary PSNR "
            "change < 0.03 dB over the final snapshot interval"
        ),
        "snapshot_interval": int(final["iteration"] - previous["iteration"]),
        "foreground_psnr_gain_db": float(fg_gain),
        "boundary_psnr_gain_db": float(boundary_gain),
        "foreground_p99_error_reduction": float(p99_drop),
    }


def _run_arm(spec: ArmSpec, args: argparse.Namespace, initial_path: Path, target, mask,
             boundary, out: Path) -> dict[str, Any]:
    from structsplat.fit import fit
    from structsplat.gaussians import GaussianField

    arm_out = out / spec.key
    arm_out.mkdir(parents=True, exist_ok=False)
    field = GaussianField.load(str(initial_path), device=args.device)
    triage_cfg = _triage_config(args, spec)
    settle_cfg = _settle_config(args)
    writer = ProgressWriter(
        arm_out,
        spec.label,
        target,
        mask,
        boundary,
        args.preview_width,
        args.hole_coverage_tau,
    )
    writer.snapshot(
        field,
        triage_cfg,
        stage="initial",
        iteration=0,
        event="shared_5000_boundary_sampled",
    )

    print(
        f"\n=== {spec.label}: pooled triage {args.triage_iters} iters, "
        f"coverage={spec.coverage_target or 'off'} ===",
        flush=True,
    )
    triage_started = time.perf_counter()
    diagnostics = SpawnDiagnostics(
        arm_out, triage_cfg, target, mask, boundary, args.preview_width
    )

    def triage_observer(observed_field, iteration: int, loss: float) -> None:
        diagnostics.post_event(observed_field, iteration)
        if iteration % int(args.snapshot_every) == 0 or iteration == args.triage_iters:
            writer.snapshot(
                observed_field,
                triage_cfg,
                stage="triage",
                iteration=iteration,
                event="fit",
                loss=loss,
            )

    with diagnostics:
        triage_output = fit(
            field,
            target,
            triage_cfg,
            mask=mask.detach().cpu().numpy(),
            verbose=True,
            iteration_observer=triage_observer,
            # Every event must reach diagnostics.post_event; image snapshots are independently
            # throttled inside the observer.
            observer_every=int(args.triage_every),
        )
    triage_seconds = time.perf_counter() - triage_started
    field = triage_output["field"]
    field.save(str(arm_out / "pooled_terminal.npz"))
    pooled_terminal = writer.snapshot(
        field,
        triage_cfg,
        stage="triage",
        iteration=args.triage_iters,
        event="compacted_terminal",
    )
    _atomic_json(
        arm_out / "triage_fit.json", _fit_record(triage_output, triage_cfg)
    )

    print(
        f"\n=== {spec.label}: fixed-topology L2 settle {args.settle_iters} iters ===",
        flush=True,
    )
    settle_started = time.perf_counter()

    def settle_observer(observed_field, local_iteration: int, loss: float) -> None:
        if (
            local_iteration % int(args.snapshot_every) == 0
            or local_iteration == args.settle_iters
        ):
            writer.snapshot(
                observed_field,
                settle_cfg,
                stage="settle",
                iteration=args.triage_iters + local_iteration,
                event="fit",
                loss=loss,
            )

    settle_output = fit(
        field,
        target,
        settle_cfg,
        mask=mask.detach().cpu().numpy(),
        verbose=True,
        iteration_observer=settle_observer,
        observer_every=int(args.snapshot_every),
    )
    settle_seconds = time.perf_counter() - settle_started
    field = settle_output["field"]
    final_iteration = int(args.triage_iters + args.settle_iters)
    final = writer.snapshot(
        field,
        settle_cfg,
        stage="settle",
        iteration=final_iteration,
        event="terminal",
        native=True,
    )
    diagnostics.finalize(settle_output["render"])
    spawn_aggregate = _aggregate_spawn(diagnostics.records)
    field_path = arm_out / "final_field.npz"
    field.save(str(field_path))
    _atomic_json(arm_out / "settle_fit.json", _fit_record(settle_output, settle_cfg))
    summary = {
        "schema": "structsplat.janelle.pooled_boundary_arm.v1",
        "arm": asdict(spec),
        "initial": writer.records[0],
        "pooled_terminal": pooled_terminal,
        "final": final,
        "pool": triage_output["pool"],
        "triage_config": asdict(triage_cfg),
        "settle_config": asdict(settle_cfg),
        "triage_events": triage_output["history"]["triage_events"],
        "spawn_diagnostics": spawn_aggregate,
        "spawn_events_path": "spawn_events.json",
        "progress": writer.records,
        "convergence": _convergence_record(writer.records),
        "field": {
            "path": field_path.name,
            "sha256": _sha256(field_path),
            "n_gaussians": int(field.n),
        },
        "timing": {
            "triage_seconds": triage_seconds,
            "settle_seconds": settle_seconds,
            "total_fit_seconds": triage_seconds + settle_seconds,
        },
    }
    _atomic_json(arm_out / "summary.json", summary)
    _write_arm_index(arm_out, summary, diagnostics.records)
    return summary


def _fmt(value: Any, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "—"
        return f"{float(value):.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _sparkline(records: list[dict[str, Any]], key: str, color: str) -> str:
    if len(records) < 2:
        return ""
    values = [float(record[key]) for record in records]
    iterations = [int(record["iteration"]) for record in records]
    width, height, pad = 360, 100, 8
    xmin, xmax = min(iterations), max(iterations)
    ymin, ymax = min(values), max(values)
    yrange = max(ymax - ymin, 1e-12)
    xrange = max(xmax - xmin, 1)
    points = " ".join(
        f"{pad + (x - xmin) / xrange * (width - 2 * pad):.1f},"
        f"{height - pad - (y - ymin) / yrange * (height - 2 * pad):.1f}"
        for x, y in zip(iterations, values)
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img'>"
        f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{points}'/>"
        f"<text x='8' y='13'>{html.escape(key)} max {_fmt(ymax, 4)}</text>"
        f"<text x='8' y='{height - 4}'>min {_fmt(ymin, 4)}</text></svg>"
    )


def _write_arm_index(out: Path, summary: dict[str, Any],
                     spawn_events: list[dict[str, Any]]) -> None:
    progress_rows = []
    for record in summary["progress"]:
        progress_rows.append(
            "<tr>"
            f"<td>{record['iteration']}</td><td>{html.escape(record['stage'])}</td>"
            f"<td>{record['n_live']}</td>"
            f"<td>{record['foreground_psnr_raw_db']:.3f}</td>"
            f"<td>{record['boundary_psnr_raw_db']:.3f}</td>"
            f"<td>{record['foreground_error_p99']:.4f}</td>"
            f"<td>{100.0 * record['undercoverage_fraction']:.3f}%</td>"
            f"<td><img src='{html.escape(record['reconstruction'])}'></td>"
            f"<td><img src='{html.escape(record['absolute_error_x4'])}'></td>"
            f"<td><img src='{html.escape(record['undercoverage'])}'></td>"
            "</tr>"
        )
    spawn_rows = []
    for event in spawn_events:
        spawn_rows.append(
            "<tr>"
            f"<td>{event['event']}</td><td>{event['iteration']}</td>"
            f"<td>{event['spawned']}</td><td>{event['spawned_holes_core']}</td>"
            f"<td>{_fmt(event.get('pre_score_error_mean'))}</td>"
            f"<td>{_fmt(event.get('pre_activation_error_mean'))}</td>"
            f"<td>{_fmt(event.get('post_event_error_mean'))}</td>"
            f"<td>{_fmt(event.get('post_event_improved_fraction'), 3)}</td>"
            f"<td>{_fmt(event.get('terminal_error_mean'))}</td>"
            f"<td>{_fmt(event.get('terminal_improved_fraction'), 3)}</td>"
            + (
                f"<td><img src='{html.escape(event['selected_sites_overlay'])}'></td>"
                if event.get("selected_sites_overlay")
                else "<td>—</td>"
            )
            + "</tr>"
        )
    aggregate = summary["spawn_diagnostics"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(summary['arm']['label'])} — Janelle pooled boundary test</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#9da7b3}}
body{{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);margin:20px}}
a{{color:#58a6ff}} .note{{color:var(--muted);max-width:1200px}} .cards{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{background:var(--panel);border:1px solid var(--line);padding:12px;min-width:190px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid var(--line);padding:6px;vertical-align:top}}
th{{position:sticky;top:0;background:#21262d;z-index:1}} img{{width:360px;max-width:30vw;height:auto;background:#000}}
svg{{background:var(--panel);border:1px solid var(--line);width:360px;max-width:95vw}} text{{fill:var(--muted);font-size:10px}}
</style></head><body>
<p><a href="../index.html">← common comparison</a></p>
<h1>{html.escape(summary['arm']['label'])}</h1>
<p class="note">{html.escape(summary['arm']['description'])} Every arm starts from the same
5,000-row boundary-sampled field. Error is mean absolute RGB, visualized at a fixed 4× scale.
Red in undercoverage images means raw compositor coverage &lt; {summary['final']['coverage_tau']};
cyan marks the measured inside boundary band.</p>
<div class="cards">
<div class="card"><b>FG PSNR</b><br>{summary['final']['foreground_psnr_raw_db']:.3f} dB</div>
<div class="card"><b>Boundary PSNR</b><br>{summary['final']['boundary_psnr_raw_db']:.3f} dB</div>
<div class="card"><b>FG error p99</b><br>{summary['final']['foreground_error_p99']:.4f}</div>
<div class="card"><b>Undercoverage</b><br>{100.0 * summary['final']['undercoverage_fraction']:.3f}%</div>
<div class="card"><b>Spawned total</b><br>{aggregate.get('spawned_total', 0)}</div>
<div class="card"><b>Spawn-site terminal reduction</b><br>
{100.0 * aggregate.get('terminal_relative_reduction_at_spawn_sites', 0.0):.2f}%</div>
</div>
<h2>Trajectories</h2>
<div class="cards">
{_sparkline(summary['progress'], 'foreground_psnr_raw_db', '#58a6ff')}
{_sparkline(summary['progress'], 'boundary_psnr_raw_db', '#3fb950')}
{_sparkline(summary['progress'], 'foreground_error_p99', '#f0883e')}
{_sparkline(summary['progress'], 'undercoverage_fraction', '#f85149')}
</div>
<h2>Intermediate reconstructions, error, and coverage holes</h2>
<table><thead><tr><th>iter</th><th>stage</th><th>live N</th><th>FG PSNR</th>
<th>boundary PSNR</th><th>error p99</th><th>holes</th><th>reconstruction</th>
<th>error ×4</th><th>coverage &lt; τ</th></tr></thead>
<tbody>{''.join(progress_rows)}</tbody></table>
<h2>High-error activation audit</h2>
<p class="note">Yellow circles are non-boundary births; cyan circles are births in the measured
boundary band. “Pre activation” is a fresh render after park/merge/split but before spawn;
“post event” is after spawn and containment recertification. This is the closest local
counterfactual available without changing the fitter.</p>
<table><thead><tr><th>event</th><th>iter</th><th>spawn</th><th>core holes</th>
<th>rank-render error</th><th>pre activation</th><th>post event</th><th>improved frac</th>
<th>terminal error</th><th>terminal improved frac</th><th>selected sites</th></tr></thead>
<tbody>{''.join(spawn_rows)}</tbody></table>
</body></html>"""
    _atomic_text(out / "index.html", document)


def _historical_record(path: Path) -> dict[str, Any] | None:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    final = payload.get("full_field", {}).get("metrics")
    if final is None:
        return None
    return {
        "label": "Historical complete refinement",
        "path": path,
        "final": final,
        "iterations": payload.get("convergence", {}).get("global_iterations"),
        "scope": "historical/non-fair reference (different lifecycle and horizon)",
    }


def _write_common_index(out: Path, manifest: dict[str, Any],
                        summaries: list[dict[str, Any]],
                        historical: dict[str, Any] | None) -> None:
    metric_rows = []
    for summary in summaries:
        final = summary["final"]
        spawn = summary["spawn_diagnostics"]
        metric_rows.append(
            "<tr>"
            f"<td><a href='{summary['arm']['key']}/index.html'>{html.escape(summary['arm']['label'])}</a></td>"
            f"<td>{final['n_live']}</td>"
            f"<td>{final['foreground_psnr_raw_db']:.3f}</td>"
            f"<td>{final['boundary_psnr_raw_db']:.3f}</td>"
            f"<td>{final['foreground_mae_clamped']:.5f}</td>"
            f"<td>{final['foreground_error_p99']:.4f}</td>"
            f"<td>{final['foreground_error_top1pct_mean']:.4f}</td>"
            f"<td>{100.0 * final['undercoverage_fraction']:.3f}%</td>"
            f"<td>{100.0 * final['boundary_undercoverage_fraction']:.3f}%</td>"
            f"<td>{spawn.get('spawned_total', 0)}</td>"
            f"<td>{100.0 * spawn.get('post_event_improved_fraction', 0.0):.1f}%</td>"
            f"<td>{100.0 * spawn.get('terminal_relative_reduction_at_spawn_sites', 0.0):.1f}%</td>"
            f"<td>{_fmt(final['outside_mask_max_abs'], 8)}</td>"
            f"<td>{summary['timing']['total_fit_seconds'] / 60.0:.1f} min</td>"
            "</tr>"
        )
    if historical is not None:
        final = historical["final"]
        metric_rows.append(
            "<tr class='historical'>"
            f"<td>{html.escape(historical['label'])}<br><small>{html.escape(historical['scope'])}</small></td>"
            f"<td>{final.get('n_gaussians', '—')}</td>"
            f"<td>{_fmt(final.get('foreground_psnr_raw_db'), 3)}</td>"
            f"<td>{_fmt(final.get('boundary_psnr_raw_db'), 3)}</td>"
            f"<td>{_fmt(final.get('foreground_mae_clamped'), 5)}</td>"
            "<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>"
            f"<td>{_fmt(final.get('outside_mask_max_abs'), 8)}</td><td>—</td></tr>"
        )

    columns: list[tuple[str, str, str]] = []
    for summary in summaries:
        key = summary["arm"]["key"]
        label = summary["arm"]["label"]
        columns.append((label, f"{key}/images/final_reconstruction.png", "reconstruction"))
    if historical is not None:
        old_relative = os.path.relpath(historical["path"], out)
        native = historical["final"].get("native", {})
        if native.get("reconstruction"):
            columns.append(
                (
                    historical["label"],
                    f"{old_relative}/{native['reconstruction']}",
                    "historical reconstruction",
                )
            )
    recon_figures = "".join(
        f"<figure><figcaption>{html.escape(label)}</figcaption>"
        f"<img src='{html.escape(path)}' alt='{html.escape(alt)}'></figure>"
        for label, path, alt in columns
    )
    error_figures = "".join(
        f"<figure><figcaption>{html.escape(summary['arm']['label'])}</figcaption>"
        f"<img src='{summary['arm']['key']}/images/final_absolute_error_x4.png' alt='error x4'></figure>"
        for summary in summaries
    )
    hole_figures = "".join(
        f"<figure><figcaption>{html.escape(summary['arm']['label'])}</figcaption>"
        f"<img src='{summary['arm']['key']}/images/final_undercoverage.png' alt='undercoverage'></figure>"
        for summary in summaries
    )
    trajectory_figures = "".join(
        "<section class='trajectory'>"
        f"<h3>{html.escape(summary['arm']['label'])}</h3>"
        f"{_sparkline(summary['progress'], 'foreground_psnr_raw_db', '#58a6ff')}"
        f"{_sparkline(summary['progress'], 'boundary_psnr_raw_db', '#3fb950')}"
        f"{_sparkline(summary['progress'], 'foreground_error_p99', '#f0883e')}"
        f"{_sparkline(summary['progress'], 'undercoverage_fraction', '#f85149')}"
        "</section>"
        for summary in summaries
    )
    completed = len(summaries)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Janelle C0001 — pooled boundary variants</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#9da7b3}}
body{{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);margin:20px}}
a{{color:#58a6ff}} .note{{color:var(--muted);max-width:1250px;line-height:1.45}}
.status{{padding:10px;background:#1f2937;border-left:4px solid #58a6ff;max-width:900px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid var(--line);padding:7px;vertical-align:top}}
th{{background:#21262d;position:sticky;top:0}} .historical{{color:var(--muted)}}
.compare{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}}
figure{{margin:0;background:var(--panel);border:1px solid var(--line);padding:10px}}
figcaption{{font-weight:650;margin-bottom:7px}} figure img{{width:100%;height:auto;background:#000}}
.trajectories{{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:12px}}
.trajectory{{background:var(--panel);border:1px solid var(--line);padding:10px}}
svg{{width:100%;height:100px;background:#0d1117;margin:3px 0}} text{{fill:var(--muted);font-size:10px}}
code{{background:#21262d;padding:2px 4px}}
</style></head><body>
<h1>Janelle frame_00008 / C0001 — pooled boundary variants</h1>
<p class="status"><b>{completed}/{len(ARMS)} fair arms complete.</b> Source commit
<code>{html.escape(manifest['repository']['commit'][:12])}</code>; native fitted crop
{manifest['fit_size'][0]}×{manifest['fit_size'][1]} px; shared start N=5,000; pool capacity
{manifest['protocol']['pool_capacity']}; triage {manifest['protocol']['triage_iters']} iterations
+ fixed-topology settle {manifest['protocol']['settle_iters']} iterations.</p>
<p class="note">All fair arms use the exact same saved initialization, mask, seed, optimizer,
topology budgets, and horizon. The only arm-level change is the FIT-022 coverage target (or off).
The historical row is included only for visual context and is not a controlled comparison.
Error images use one fixed 4× mean-absolute-RGB scale. Undercoverage images mark raw normalized-
compositor coverage &lt; {manifest['protocol']['hole_coverage_tau']} in red and the 5×5 inside
mask-boundary band in cyan. Exact zero outside the authoritative mask is checked numerically.</p>
<h2>Quantitative comparison</h2>
<table><thead><tr><th>variant</th><th>final N</th><th>FG PSNR</th><th>boundary PSNR</th>
<th>FG MAE</th><th>error p99</th><th>top-1% mean</th><th>holes</th><th>boundary holes</th>
<th>spawned</th><th>spawn immediate improved</th><th>spawn-site terminal reduction</th>
<th>outside max</th><th>fit time</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table>
<h2>Final reconstructions</h2><div class="compare">{recon_figures}</div>
<h2>Final absolute error ×4</h2><div class="compare">{error_figures}</div>
<h2>Final undercoverage / holes</h2><div class="compare">{hole_figures}</div>
<h2>Trajectories</h2><div class="trajectories">{trajectory_figures}</div>
<h2>Interpretation guardrails</h2>
<p class="note">This is a source-bound, one-image development screen. It can falsify a mechanism
on this image and expose boundary leakage or failed births, but it cannot promote a repository
default or support a dataset-level quality claim. Per-event coordinates and local before/after
errors are available in each arm's <code>spawn_events.json</code>; full configs, histories,
source snapshots, hashes, and native fields are retained alongside this page.</p>
</body></html>"""
    _atomic_text(out / "index.html", document)


def _selected_arms(text: str) -> list[ArmSpec]:
    requested = [part.strip() for part in text.split(",") if part.strip()]
    if not requested or requested == ["all"]:
        return list(ARMS)
    by_key = {arm.key: arm for arm in ARMS}
    unknown = [key for key in requested if key not in by_key]
    if unknown:
        raise ValueError(
            f"unknown arms {unknown}; choose from {sorted(by_key)} or 'all'"
        )
    return [by_key[key] for key in requested]


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this native-resolution comparison requires a CUDA device")
    torch.cuda.set_device(device)
    torch.manual_seed(int(args.torch_seed))
    np.random.seed(int(args.torch_seed))
    torch.cuda.reset_peak_memory_stats(device)
    bridge = _load_bridge(args.realtime_root.resolve())
    source = _find_source(
        bridge, args.capture_root.resolve(), args.frame, args.view_id
    )
    prepared = bridge.CONVERTER.prepare_source_view(source)
    if prepared.alpha_crop is None:
        raise RuntimeError("prepared source has no authoritative mask")
    mask_cpu = prepared.alpha_crop.bool().contiguous()
    mask = mask_cpu.to(device)
    target = (
        prepared.rgb * mask_cpu[..., None].to(prepared.rgb)
    ).to(device).contiguous()
    height, width = mask.shape
    outside = (~mask).to(target).view(1, 1, height, width)
    near_outside = F.max_pool2d(outside, kernel_size=5, stride=1, padding=2)[0, 0] > 0
    boundary = mask & near_outside

    shared = out / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    _to_image(target).save(shared / "target.png")
    Image.fromarray(
        (mask_cpu.numpy().astype(np.uint8) * 255), mode="L"
    ).save(shared / "authoritative_mask.png")
    field, _, init_cfg, tensor_cfg, initial_render_cfg, boundary_added = _initial_field(
        target, mask, source.seed, device, float(args.mask_margin)
    )
    initial_path = shared / "initial_5000_boundary_sampled.npz"
    field.save(str(initial_path))
    initial_render, initial_den, initial_metrics = _evaluate(
        field,
        initial_render_cfg,
        target,
        mask,
        boundary,
        float(args.hole_coverage_tau),
    )
    mask3 = mask[..., None].to(target)
    _to_image(initial_render.clamp(0.0, 1.0) * mask3).save(
        shared / "initial_reconstruction.png"
    )
    initial_error = (
        initial_render.clamp(0.0, 1.0) - target
    ).abs().mean(dim=2) * mask
    _to_image(_error_heatmap(initial_error)).save(shared / "initial_absolute_error_x4.png")
    _to_image(
        _hole_visual(
            target, initial_den, mask, boundary, float(args.hole_coverage_tau)
        )
    ).save(shared / "initial_undercoverage.png")
    del field, initial_render, initial_den
    torch.cuda.empty_cache()

    snapshots = _source_snapshots(out)
    manifest = {
        "schema": "structsplat.janelle.pooled_boundary_comparison.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": bridge._source_record(source),
        "fit_window": list(prepared.fit_window),
        "fit_size": [width, height],
        "foreground_pixels": int(mask.sum()),
        "boundary_pixels": int(boundary.sum()),
        "shared_initialization": {
            "path": str(initial_path.relative_to(out)),
            "sha256": _sha256(initial_path),
            "n_gaussians": 5_000,
            "general_quadtree_wse": 4_500,
            "boundary_residual_tangent": int(boundary_added),
            "init_config": asdict(init_cfg),
            "structure_tensor_config": asdict(tensor_cfg),
            "metrics": initial_metrics,
        },
        "protocol": {
            "arms": [asdict(arm) for arm in _selected_arms(args.arms)],
            "pool_capacity": int(args.pool_capacity),
            "triage_iters": int(args.triage_iters),
            "settle_iters": int(args.settle_iters),
            "triage_every": int(args.triage_every),
            "snapshot_every": int(args.snapshot_every),
            "triage_spawn_count": int(args.triage_spawn_count),
            "triage_split_count": int(args.triage_split_count),
            "triage_merge_count": int(args.triage_merge_count),
            "triage_park_count": int(args.triage_park_count),
            "coverage_weight": float(args.coverage_weight),
            "coverage_decay_frac": float(args.coverage_decay_frac),
            "hole_coverage_tau": float(args.hole_coverage_tau),
            "torch_seed": int(args.torch_seed),
            "fairness": (
                "same saved 5k start, mask, seed, capacity, topology budgets, optimizer, "
                "iteration horizon, and terminal settle; only coverage target/off differs"
            ),
        },
        "repository": {
            **_git_record(),
            "executed_source_snapshots": snapshots,
            "python_import_root": str((REPOSITORY_ROOT / "src").resolve()),
        },
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
    }
    _atomic_json(out / "run_manifest.json", manifest)
    historical = _historical_record(args.historical_run.resolve())
    summaries: list[dict[str, Any]] = []
    _write_common_index(out, manifest, summaries, historical)
    for spec in _selected_arms(args.arms):
        torch.manual_seed(int(args.torch_seed))
        np.random.seed(int(args.torch_seed))
        summary = _run_arm(
            spec, args, initial_path, target, mask, boundary, out
        )
        summaries.append(summary)
        _atomic_json(
            out / "comparison.json",
            {
                "schema": "structsplat.janelle.pooled_boundary_results.v1",
                "manifest": "run_manifest.json",
                "arms": summaries,
                "historical": historical,
            },
        )
        _write_common_index(out, manifest, summaries, historical)
        torch.cuda.empty_cache()
    final_record = {
        "schema": "structsplat.janelle.pooled_boundary_results.v1",
        "manifest": "run_manifest.json",
        "arms": summaries,
        "historical": historical,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    _atomic_json(out / "comparison.json", final_record)
    _write_common_index(out, manifest, summaries, historical)
    print(f"\nComparison complete: {out / 'index.html'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--realtime-root", type=Path, default=DEFAULT_REALTIME_ROOT)
    parser.add_argument("--frame", default="frame_00008")
    parser.add_argument("--view-id", default="C0001")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--historical-run", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arms", default="all")
    parser.add_argument("--mask-margin", type=float, default=0.75)
    parser.add_argument("--pool-capacity", type=int, default=11_000)
    parser.add_argument("--triage-iters", type=int, default=8_000)
    parser.add_argument("--settle-iters", type=int, default=6_000)
    parser.add_argument("--triage-every", type=int, default=500)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument("--triage-spawn-count", type=int, default=512)
    parser.add_argument("--triage-split-count", type=int, default=32)
    parser.add_argument("--triage-merge-count", type=int, default=64)
    parser.add_argument("--triage-park-count", type=int, default=64)
    parser.add_argument("--coverage-weight", type=float, default=0.05)
    parser.add_argument("--coverage-decay-frac", type=float, default=0.75)
    parser.add_argument("--hole-coverage-tau", type=float, default=HOLE_COVERAGE_TAU)
    parser.add_argument("--preview-width", type=int, default=1_000)
    parser.add_argument("--torch-seed", type=int, default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
