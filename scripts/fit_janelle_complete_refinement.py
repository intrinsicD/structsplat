#!/usr/bin/env python3
"""Complete boundary-aware refinement run for Janelle frame_00008/C0001.

This runner is intentionally separate from the capped multi-view production bridge.  It keeps a
full native GaussianField, exercises complementary densification operators in stages, saves
native checkpoints and visual progress, and only then derives a byte-capped realtime-gs archive.

The lifecycle is:

1. 4,500 feature-aware quadtree-WSE rows + 500 explicit boundary-residual rows (5,000 total),
2. high-error residual-tensor births + boundary births + prune + relocation,
3. count-neutral redundant-pair merge + high-error teleport,
4. moment-preserving owner splits + boundary births + prune + relocation,
5. another merge/teleport and high-error polishing stage,
6. final merge/teleport and fixed-count settle rounds until foreground and boundary PSNR plateau.

The coupled merge/teleport is a between-fit operation because StructSplat's FitConfig currently
has no merge operator. For every accepted pair, one row is enlarged at the exact center midpoint
with a covariance envelope covering both translated inputs, and the other row is teleported to a
high-error site; count is conserved. Pairs whose containment cap cannot preserve the enlargement
are rejected in a batch. The operation is recorded separately and never conflated with pruning or
byte-cap backoff.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE_ROOT = Path(
    "/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric"
)
DEFAULT_REALTIME_ROOT = Path("/home/alex/Documents/realtime-gs")
DEFAULT_OUT = REPOSITORY_ROOT / "runs/janelle_C0001_complete_refinement_20260722"


def _prepend_source_roots(realtime_root: Path) -> None:
    for root in reversed((REPOSITORY_ROOT / "src", realtime_root / "src")):
        text = str(root.resolve())
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def _load_bridge(realtime_root: Path):
    os.environ["STRUCTSPLAT_REALTIME_ROOT"] = str(realtime_root.resolve())
    scripts_root = str((REPOSITORY_ROOT / "scripts").resolve())
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    import fit_janelle_mask_contained as bridge

    return bridge


@dataclass(frozen=True)
class StageSpec:
    name: str
    mode: str
    max_gaussians: int
    iters: int
    split_every: int | None
    split_count: int
    boundary_every: int | None
    boundary_count: int
    early_stop_min_iters: int
    merge_after: int = 0


GROWTH_STAGES = (
    StageSpec(
        name="high_error_growth",
        mode="high_error_birth",
        max_gaussians=8_000,
        iters=8_000,
        split_every=500,
        split_count=256,
        boundary_every=500,
        boundary_count=128,
        early_stop_min_iters=5_000,
        merge_after=128,
    ),
    StageSpec(
        name="moment_split_growth",
        mode="moment_split",
        max_gaussians=10_000,
        iters=8_000,
        split_every=500,
        split_count=256,
        boundary_every=500,
        boundary_count=96,
        early_stop_min_iters=4_500,
        merge_after=256,
    ),
    StageSpec(
        name="high_error_polish",
        mode="high_error_birth",
        max_gaussians=11_000,
        iters=7_000,
        split_every=500,
        split_count=128,
        boundary_every=500,
        boundary_count=128,
        early_stop_min_iters=3_500,
        merge_after=128,
    ),
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _git_record() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(REPOSITORY_ROOT), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    status = git("status", "--short")
    diff = git("diff", "--binary", "--", "src", "scripts", "README.md", "tests")
    return {
        "commit": git("rev-parse", "HEAD"),
        "status": status,
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "relevant_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def _snapshot_executed_sources(out: Path) -> list[dict[str, Any]]:
    """Persist dirty/untracked orchestration sources so a run is source-bound, not hash-only."""
    sources = (
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "scripts/fit_janelle_mask_contained.py",
    )
    records = []
    for source in sources:
        payload = source.read_bytes()
        relative = Path("provenance") / source.name
        _atomic_text(out / relative, payload.decode("utf-8"))
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


def _resize_hwc(value, width: int):
    import torch.nn.functional as F

    if value.shape[1] <= width:
        return value
    height = max(1, round(value.shape[0] * width / value.shape[1]))
    return F.interpolate(
        value.permute(2, 0, 1).unsqueeze(0),
        size=(height, width),
        mode="area",
    )[0].permute(1, 2, 0)


def _to_image(value) -> Image.Image:
    array = np.rint(value.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _error_heatmap(error):
    x = (error * 4.0).clamp(0.0, 1.0)
    red = (2.2 * x - 0.35).clamp(0.0, 1.0)
    green = (2.5 * x - 1.3).clamp(0.0, 1.0)
    blue = (1.8 * x).clamp(0.0, 1.0) * (1.0 - 0.65 * x)
    import torch

    return torch.stack([red, green, blue], dim=-1)


def _render_field(field, cfg, height: int, width: int):
    from structsplat.render import render_field

    return render_field(
        field.means,
        field.conics(cfg.aa_dilation),
        field.colors,
        field.radii(cfg.sigma_cutoff, cfg.aa_dilation),
        height,
        width,
        chunk=cfg.render_chunk,
        mode=cfg.renderer,
        opacities=field.opacity_values(),
        scales=field.effective_scales(cfg.aa_dilation),
        rotations=field.rotations,
        support_fade=cfg.support_fade,
        support_fade_alpha=1.0,
        sigma_cutoff=cfg.sigma_cutoff,
        color_grads=field.color_grads,
    )


def _metric_record(render, target, mask, boundary) -> dict[str, Any]:
    import torch

    error = render - target
    clamped = render.clamp(0.0, 1.0)
    clamped_error = clamped - target
    foreground_mse = float(error[mask].square().mean())
    foreground_clamped_mse = float(clamped_error[mask].square().mean())
    boundary_mse = float(error[boundary].square().mean())
    outside = render[~mask].abs()
    return {
        "foreground_psnr_raw_db": _psnr(foreground_mse),
        "foreground_psnr_clamped_db": _psnr(foreground_clamped_mse),
        "foreground_mae_clamped": float(clamped_error[mask].abs().mean()),
        "boundary_psnr_raw_db": _psnr(boundary_mse),
        "boundary_mae_clamped": float(clamped_error[boundary].abs().mean()),
        "matted_crop_psnr_raw_db": _psnr(float(error.square().mean())),
        "outside_mask_max_abs": float(outside.max()) if outside.numel() else 0.0,
        "outside_mask_mean_abs": float(outside.mean()) if outside.numel() else 0.0,
        "outside_mask_exact_zero": bool(not outside.numel() or float(outside.max()) == 0.0),
        "render_out_of_range_fraction": float(((render < 0.0) | (render > 1.0)).float().mean()),
        "finite": bool(torch.isfinite(render).all()),
    }


class ProgressWriter:
    def __init__(self, out: Path, target, mask, boundary, cfg, preview_width: int):
        self.out = out
        self.target = target
        self.mask = mask
        self.mask3 = mask[..., None].to(target)
        self.boundary = boundary
        self.cfg = cfg
        self.preview_width = int(preview_width)
        self.height, self.width = mask.shape
        self.records: list[dict[str, Any]] = []
        self.started = time.perf_counter()
        target_preview = _resize_hwc(target * self.mask3, self.preview_width)
        target_path = self.out / "images/target_foreground.jpg"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _to_image(target_preview).save(target_path, quality=94, subsampling=0)

    def snapshot(
        self,
        field,
        *,
        stage: str,
        global_iter: int,
        event: str,
        loss: float | None = None,
        native: bool = False,
    ) -> dict[str, Any]:
        import torch

        with torch.no_grad():
            render = _render_field(field, self.cfg, self.height, self.width)
            metrics = _metric_record(render, self.target, self.mask, self.boundary)
            display_render = render.clamp(0.0, 1.0) * self.mask3
            display_error = (render.clamp(0.0, 1.0) - self.target).abs().mean(dim=-1)
            display_error = display_error * self.mask
            recon_preview = _resize_hwc(display_render, self.preview_width)
            error_preview = _resize_hwc(_error_heatmap(display_error), self.preview_width)

        ordinal = len(self.records)
        stem = f"{ordinal:03d}_iter_{int(global_iter):06d}_{stage}_{event}"
        recon_rel = Path("images/intermediates") / f"{stem}_reconstruction.jpg"
        error_rel = Path("images/intermediates") / f"{stem}_absolute_error_x4.jpg"
        (self.out / recon_rel).parent.mkdir(parents=True, exist_ok=True)
        _to_image(recon_preview).save(self.out / recon_rel, quality=92, subsampling=0)
        _to_image(error_preview).save(self.out / error_rel, quality=92, subsampling=0)
        native_paths = None
        if native:
            native_recon = Path("images/native") / f"{stem}_reconstruction.png"
            native_error = Path("images/native") / f"{stem}_absolute_error_x4.png"
            (self.out / native_recon).parent.mkdir(parents=True, exist_ok=True)
            _to_image(display_render).save(self.out / native_recon)
            _to_image(_error_heatmap(display_error)).save(self.out / native_error)
            native_paths = {
                "reconstruction": native_recon.as_posix(),
                "absolute_error_x4": native_error.as_posix(),
            }
        record = {
            "ordinal": ordinal,
            "stage": stage,
            "event": event,
            "global_iter": int(global_iter),
            "n_gaussians": int(field.n),
            "loss": None if loss is None else float(loss),
            "elapsed_seconds": time.perf_counter() - self.started,
            **metrics,
            "reconstruction": recon_rel.as_posix(),
            "absolute_error_x4": error_rel.as_posix(),
            "native": native_paths,
        }
        self.records.append(record)
        self.flush()
        print(
            f"[snapshot {ordinal:03d}] {stage}/{event} iter={global_iter} n={field.n} "
            f"fg={metrics['foreground_psnr_raw_db']:.3f}dB "
            f"boundary={metrics['boundary_psnr_raw_db']:.3f}dB",
            flush=True,
        )
        return record

    def flush(self) -> None:
        _atomic_json(self.out / "progress.json", {"snapshots": self.records})
        if self.records:
            fieldnames = list(self.records[0])
            temporary = self.out / f".progress.csv.tmp-{os.getpid()}"
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in self.records:
                    writer.writerow({key: row.get(key) for key in fieldnames})
            os.replace(temporary, self.out / "progress.csv")
        rows = []
        for row in self.records:
            rows.append(
                "<tr>"
                f"<td>{row['ordinal']}</td>"
                f"<td>{html.escape(row['stage'])}<br>{html.escape(row['event'])}</td>"
                f"<td>{row['global_iter']}</td><td>{row['n_gaussians']}</td>"
                f"<td>{row['foreground_psnr_raw_db']:.3f}</td>"
                f"<td>{row['boundary_psnr_raw_db']:.3f}</td>"
                f"<td><img src='{html.escape(row['reconstruction'])}'></td>"
                f"<td><img src='{html.escape(row['absolute_error_x4'])}'></td>"
                "</tr>"
            )
        document = """<!doctype html><meta charset='utf-8'>
<title>Janelle C0001 complete refinement</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:20px}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #444;padding:6px;vertical-align:top}
th{position:sticky;top:0;background:#222}img{width:520px;max-width:42vw;height:auto;background:#000}
.note{max-width:1100px;color:#ccc}
</style>
<h1>Janelle frame_00008 / C0001 — complete refinement</h1>
<p class='note'>Reconstruction and fixed-scale (4× mean absolute RGB) error snapshots. Metrics are
computed at native fitted-window resolution before JPEG encoding. Boundary PSNR uses inside-mask
pixels within a 5×5 dilation of the outside.</p>
<table><thead><tr><th>#</th><th>stage/event</th><th>global iter</th><th>N</th>
<th>FG PSNR</th><th>boundary PSNR</th><th>reconstruction</th><th>error ×4</th></tr></thead>
<tbody>""" + "\n".join(rows) + "</tbody></table>\n"
        _atomic_text(self.out / "index.html", document)


def _covariances(field):
    import torch

    scales = field.scales().detach()
    c = torch.cos(field.rotations.detach())
    s = torch.sin(field.rotations.detach())
    r00, r01, r10, r11 = c, -s, s, c
    sx2, sy2 = scales[:, 0].square(), scales[:, 1].square()
    xx = r00.square() * sx2 + r01.square() * sy2
    xy = r00 * r10 * sx2 + r01 * r11 * sy2
    yy = r10.square() * sx2 + r11.square() * sy2
    return torch.stack(
        [torch.stack([xx, xy], dim=-1), torch.stack([xy, yy], dim=-1)], dim=-2
    )


def merge_redundant(
    field,
    max_pairs: int,
    mask_constraint,
    cfg,
    target,
    render_img,
    mask,
) -> tuple[Any, dict[str, Any]]:
    """Batch count-neutral merge/teleport of mutually-nearest redundant pairs.

    For each selected pair, the lower-index row becomes the merged row. Its center is the exact
    midpoint, and its covariance is the smallest scaled pair moment that envelopes both translated
    input covariances, with a further explicit enlargement. The other row is retained and
    batch-teleported to a spaced high-error pixel with local image-gradient covariance. Pairs whose
    containment cap would undo that envelope are batch-restored. No per-pair tensor mutation or
    count change occurs.
    """
    import torch
    import torch.nn.functional as F
    from scipy.spatial import cKDTree
    from structsplat.gaussians import GaussianField
    from structsplat.fit import _image_gradients

    requested = max(0, min(int(max_pairs), field.n // 4))
    if requested == 0 or field.n < 2:
        return field, {
            "requested": requested,
            "merged": 0,
            "teleported": 0,
            "candidates": 0,
            "n_before": int(field.n),
            "n_after": int(field.n),
        }
    means_cpu = field.means.detach().cpu().numpy()
    distances, neighbours = cKDTree(means_cpu).query(means_cpu, k=2, workers=-1)
    nearest = np.asarray(neighbours[:, 1], dtype=np.int64)
    indices = np.arange(field.n, dtype=np.int64)
    # Mutual nearest-neighbour pairs are disjoint by construction, eliminating a greedy Python
    # pair-selection loop while retaining deterministic pair identities.
    mutual = nearest[nearest] == indices
    first_np = indices[mutual & (indices < nearest)]
    second_np = nearest[first_np]
    if first_np.size == 0:
        return field, {
            "requested": requested,
            "merged": 0,
            "teleported": 0,
            "candidates": 0,
            "n_before": int(field.n),
            "n_after": int(field.n),
        }

    device = field.means.device
    first = torch.as_tensor(first_np, device=device, dtype=torch.long)
    second = torch.as_tensor(second_np, device=device, dtype=torch.long)
    distance = torch.as_tensor(distances[first_np, 1], device=device, dtype=field.means.dtype)
    color_delta = torch.linalg.norm(
        field.colors.detach()[first] - field.colors.detach()[second], dim=1
    )
    sorted_scales = torch.sort(field.scales().detach(), dim=1).values.clamp_min(1e-6)
    scale_delta = torch.linalg.norm(
        torch.log(sorted_scales[first]) - torch.log(sorted_scales[second]), dim=1
    )
    angle_delta = field.rotations.detach()[first] - field.rotations.detach()[second]
    axial_delta = 0.5 * torch.abs(torch.atan2(torch.sin(2 * angle_delta), torch.cos(2 * angle_delta)))
    minor = 0.5 * (sorted_scales[first, 0] + sorted_scales[second, 0])
    allowed_distance = torch.clamp(2.5 * minor, min=1.25, max=4.0)
    midpoint = 0.5 * (
        field.means.detach()[first] + field.means.detach()[second]
    )
    midpoint_x = torch.round(midpoint[:, 0]).long().clamp(0, mask.shape[1] - 1)
    midpoint_y = torch.round(midpoint[:, 1]).long().clamp(0, mask.shape[0] - 1)
    midpoint_admissible = mask_constraint.eroded_flat[
        midpoint_y * mask.shape[1] + midpoint_x
    ]
    valid = (
        (distance <= allowed_distance)
        & (color_delta <= 0.18)
        & (scale_delta <= 1.0)
        & (axial_delta <= 0.65)
        & midpoint_admissible
    )
    score = (
        distance / allowed_distance.clamp_min(1e-6)
        + 2.0 * color_delta
        + 0.25 * scale_delta
        + 0.15 * axial_delta
    )
    valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
    selected_count = min(requested, int(valid_idx.numel()))
    if selected_count <= 0:
        return field, {
            "requested": requested,
            "merged": 0,
            "teleported": 0,
            "candidates": int(valid.sum()),
            "n_before": int(field.n),
            "n_after": int(field.n),
        }
    selected_local = valid_idx[torch.argsort(score[valid_idx])[:selected_count]]
    keep_idx = first[selected_local]
    teleport_idx = second[selected_local]
    means = field.means.detach().clone()
    log_scales = field.log_scales.detach().clone()
    rotations = field.rotations.detach().clone()
    colors = field.colors.detach().clone()
    color_grads = None if field.color_grads is None else field.color_grads.detach().clone()
    opacities = None if field.opacities is None else field.opacities.detach().clone()
    amplitudes = (
        torch.ones(field.n, device=device, dtype=means.dtype)
        if opacities is None
        else torch.sigmoid(opacities)
    )
    wa, wb = amplitudes[keep_idx], amplitudes[teleport_idx]
    total = (wa + wb).clamp_min(1e-6)
    merged_mean = 0.5 * (means[keep_idx] + means[teleport_idx])
    cov = _covariances(field)
    da = means[keep_idx] - merged_mean
    db = means[teleport_idx] - merged_mean
    spatial_a = cov[keep_idx] + da[:, :, None] * da[:, None, :]
    spatial_b = cov[teleport_idx] + db[:, :, None] * db[:, None, :]
    moment_cov = 0.5 * (spatial_a + spatial_b)
    # The equal-weight moment alone can be smaller than the larger input covariance. Scale that
    # moment by the smallest common factor whose ellipse contains both translated input ellipses,
    # then add the requested 5% enlargement. This is a vectorized generalized-eigen envelope.
    moment_values, moment_vectors = torch.linalg.eigh(moment_cov)
    inv_sqrt = (
        moment_vectors
        @ torch.diag_embed(moment_values.clamp_min(1e-8).rsqrt())
        @ moment_vectors.transpose(1, 2)
    )
    normalized_a = inv_sqrt @ spatial_a @ inv_sqrt
    normalized_b = inv_sqrt @ spatial_b @ inv_sqrt
    envelope_factor = torch.maximum(
        torch.linalg.eigvalsh(normalized_a)[:, -1],
        torch.linalg.eigvalsh(normalized_b)[:, -1],
    ).clamp_min(1.0)
    merged_cov = moment_cov * envelope_factor[:, None, None] * (1.05**2)
    eigenvalues, eigenvectors = torch.linalg.eigh(merged_cov)
    eigenvalues = eigenvalues.clamp_min(0.35**2)
    major = eigenvectors[:, :, 1]
    merged_scales = torch.stack(
        [torch.sqrt(eigenvalues[:, 1]), torch.sqrt(eigenvalues[:, 0])], dim=1
    )
    means[keep_idx] = merged_mean
    log_scales[keep_idx] = torch.log(merged_scales)
    rotations[keep_idx] = torch.atan2(major[:, 1], major[:, 0])
    colors[keep_idx] = (
        wa[:, None] * colors[keep_idx] + wb[:, None] * colors[teleport_idx]
    ) / total[:, None]
    if color_grads is not None:
        color_grads[keep_idx] = (
            wa[:, None, None] * color_grads[keep_idx]
            + wb[:, None, None] * color_grads[teleport_idx]
        ) / total[:, None, None]
    if opacities is not None:
        combined = (1.0 - (1.0 - wa) * (1.0 - wb)).clamp(1e-4, 1.0 - 1e-4)
        opacities[keep_idx] = torch.logit(combined)
    else:
        opacities = torch.full((field.n,), 10.0, device=device, dtype=means.dtype)

    # High-error teleport sites are selected in one max-pool NMS + top-k batch. Restricting the
    # score to the foreground avoids wasting a freed row on the exactly-zero exterior.
    residual_map = (render_img.detach() - target.detach()).abs().mean(dim=2)
    score_map = torch.where(mask, residual_map, torch.full_like(residual_map, -float("inf")))
    pooled = F.max_pool2d(score_map[None, None], kernel_size=7, stride=1, padding=3)[0, 0]
    peak_scores = torch.where(
        mask & (score_map >= pooled), score_map, torch.full_like(score_map, -float("inf"))
    ).reshape(-1)
    finite_peaks = int(torch.isfinite(peak_scores).sum())
    if finite_peaks >= selected_count:
        sites = torch.topk(peak_scores, k=selected_count).indices
    else:
        sites = torch.topk(score_map.reshape(-1), k=selected_count).indices
    height, width = mask.shape
    site_y = torch.div(sites, width, rounding_mode="floor")
    site_x = sites - site_y * width
    means[teleport_idx] = torch.stack(
        [site_x.to(means.dtype), site_y.to(means.dtype)], dim=1
    )

    # Match the production high-error birth geometry: tangent orientation and anisotropy come
    # from the local target structure; the site ranking itself comes from current reconstruction
    # error. The entire teleported cohort is assigned without a per-row loop.
    gx, gy = _image_gradients(target)
    gradient = torch.sqrt(gx[site_y, site_x].square() + gy[site_y, site_x].square())
    gradient_ref = torch.quantile(torch.sqrt(gx.square() + gy.square()).reshape(-1), 0.95)
    coherence = torch.clamp(gradient / gradient_ref.clamp_min(1e-6), 0.0, 1.0)
    ratio = 1.0 + (cfg.densify_max_axis_ratio - 1.0) * (
        coherence ** cfg.densify_coherence_power
    )
    base_scale = max(
        math.sqrt((height * width) / max(field.n, 1)) * cfg.split_scale,
        0.35,
    )
    along = base_scale * torch.sqrt(ratio)
    across = base_scale / torch.sqrt(ratio)
    log_scales[teleport_idx] = torch.log(
        torch.stack([along, across], dim=1).clamp_min(0.35)
    )
    rotations[teleport_idx] = torch.atan2(gy[site_y, site_x], gx[site_y, site_x]) + math.pi * 0.5
    colors[teleport_idx] = target[site_y, site_x]
    opacities[teleport_idx] = torch.logit(
        torch.full(
            (selected_count,),
            cfg.relocate_init_opacity,
            device=device,
            dtype=means.dtype,
        )
    )
    if color_grads is not None:
        color_grads[teleport_idx] = 0.0
    filter_variance = (
        None if field.filter_variance is None else field.filter_variance.detach().clone()
    )
    if filter_variance is not None:
        filter_variance[keep_idx] = (
            wa * filter_variance[keep_idx] + wb * filter_variance[teleport_idx]
        ) / total
        filter_variance[teleport_idx] = 0.0
    scale_max = None if field.scale_max is None else field.scale_max.detach().clone()
    if scale_max is not None:
        scale_max[keep_idx] = float("inf")
        scale_max[teleport_idx] = float("inf")
    rewritten = GaussianField(
        means,
        log_scales,
        rotations,
        colors,
        opacities,
        scale_max,
        color_grads,
        None if field.background_mask is None else field.background_mask.detach().clone(),
        filter_variance,
    )
    mask_constraint.apply(rewritten, cfg, refresh=True)
    # Containment is non-negotiable. At a narrow midpoint its certified cap can undo the envelope,
    # so reject those pairs as a batch and restore both original rows. Successful merges therefore
    # guarantee both exact midpoint placement and an effective covariance envelope after capping.
    certified_cov = _covariances(rewritten)[keep_idx]
    certified_values, certified_vectors = torch.linalg.eigh(certified_cov)
    certified_inv_sqrt = (
        certified_vectors
        @ torch.diag_embed(certified_values.clamp_min(1e-8).rsqrt())
        @ certified_vectors.transpose(1, 2)
    )
    certified_a = certified_inv_sqrt @ spatial_a @ certified_inv_sqrt
    certified_b = certified_inv_sqrt @ spatial_b @ certified_inv_sqrt
    containment_ratio = torch.maximum(
        torch.linalg.eigvalsh(certified_a)[:, -1],
        torch.linalg.eigvalsh(certified_b)[:, -1],
    )
    envelope_kept = containment_ratio <= 1.0 + 1e-4
    containment_rejected = int((~envelope_kept).sum())
    if containment_rejected:
        restore_idx = torch.cat(
            [keep_idx[~envelope_kept], teleport_idx[~envelope_kept]], dim=0
        )
        rewritten.means[restore_idx] = field.means.detach()[restore_idx]
        rewritten.log_scales[restore_idx] = field.log_scales.detach()[restore_idx]
        rewritten.rotations[restore_idx] = field.rotations.detach()[restore_idx]
        rewritten.colors[restore_idx] = field.colors.detach()[restore_idx]
        if rewritten.opacities is not None:
            if field.opacities is None:
                rewritten.opacities[restore_idx] = 10.0
            else:
                rewritten.opacities[restore_idx] = field.opacities.detach()[restore_idx]
        if rewritten.scale_max is not None and field.scale_max is not None:
            rewritten.scale_max[restore_idx] = field.scale_max.detach()[restore_idx]
        if rewritten.color_grads is not None and field.color_grads is not None:
            rewritten.color_grads[restore_idx] = field.color_grads.detach()[restore_idx]
        if rewritten.filter_variance is not None and field.filter_variance is not None:
            rewritten.filter_variance[restore_idx] = field.filter_variance.detach()[restore_idx]
    keep_idx = keep_idx[envelope_kept]
    teleport_idx = teleport_idx[envelope_kept]
    site_y = site_y[envelope_kept]
    site_x = site_x[envelope_kept]
    selected_count = int(envelope_kept.sum())
    if selected_count == 0:
        return field, {
            "requested": requested,
            "merged": 0,
            "teleported": 0,
            "candidates": int(valid.sum()),
            "selected_before_containment": int(envelope_kept.numel()),
            "containment_rejected": containment_rejected,
            "n_before": int(field.n),
            "n_after": int(field.n),
            "count_conserved": True,
        }
    stats = {
        "requested": requested,
        "merged": selected_count,
        "teleported": selected_count,
        "candidates": int(valid.sum()),
        "selected_before_containment": int(envelope_kept.numel()),
        "containment_rejected": containment_rejected,
        "n_before": int(field.n),
        "n_after": int(rewritten.n),
        "mean_pair_distance_px": float(
            torch.linalg.norm(
                field.means.detach()[keep_idx] - field.means.detach()[teleport_idx], dim=1
            ).mean()
        ),
        "mean_pair_color_distance": float(
            torch.linalg.norm(
                field.colors.detach()[keep_idx] - field.colors.detach()[teleport_idx], dim=1
            ).mean()
        ),
        "mean_teleport_residual": float(residual_map[site_y, site_x].mean()),
        "pair_selection": "batched mutual-nearest + distance/color/covariance/orientation gates",
        "merge_rule": (
            "exact midpoint + minimal generalized-eigen covariance envelope of both translated "
            "inputs + 1.05 enlargement; batch reject if containment would shrink the envelope"
        ),
        "teleport_rule": "batched 7x7 residual-peak NMS + local target-gradient covariance + low-opacity target color",
        "count_conserved": int(rewritten.n) == int(field.n),
    }
    return rewritten, stats


def _fit_config(stage: StageSpec, *, margin: float):
    from structsplat.config import FitConfig

    kwargs: dict[str, Any] = {
        "iters": stage.iters,
        "renderer": "cuda_tiled",
        "render_chunk": 512,
        "pixel_loss": "l1",
        "ssim_weight": 0.3,
        "ssim_backend": "builtin",
        "loss_weighting": "mask",
        "mask_contain": True,
        "mask_margin": margin,
        "mask_cap_mode": "anisotropic",
        "mask_cap_refresh_every": 100,
        "mask_undercoverage_weight": 0.0,
        "mask_boundary_add_every": stage.boundary_every,
        "mask_boundary_add_count": stage.boundary_count,
        "mask_boundary_add_band": 4.0,
        "mask_boundary_add_spacing": 3.0,
        "support_fade": True,
        "log_every": 100,
        "checkpoint_policy": "best_psnr_final_count",
        "split_every": stage.split_every,
        "split_count": stage.split_count,
        "split_scale": 0.7 if stage.mode == "moment_split" else 0.35,
        "split_oversample": 8.0,
        "split_min_spacing": 1.0,
        "split_color_init": "target",
        "seed_new_row_optimizer_state": False,
        "new_row_temper_iters": 0,
        "prune_every": 1_000,
        "prune_min_activity": 1e-2,
        "prune_keep_min": 4_000,
        "relocate_every": 750,
        "relocate_at_split": False,
        "relocate_count": 128,
        "relocate_init_opacity": 0.05,
        "relocate_residual_downsample": 4,
        "max_gaussians": stage.max_gaussians,
        "split_schedule_stops_at_max": True,
        "early_stop_patience": 12,
        "early_stop_min_delta": 0.005,
        "early_stop_min_iters": stage.early_stop_min_iters,
        "compute_lpips": False,
    }
    if stage.mode == "high_error_birth":
        kwargs.update(
            refine_site="residual_tensor",
            refine_primitive="sampled_add",
            refine_nms="on",
        )
    elif stage.mode == "moment_split":
        kwargs.update(
            refine_site="support",
            refine_primitive="moment_preserving",
            refine_nms="off",
        )
    elif stage.mode == "settle":
        kwargs.update(
            # The growth objective is deliberately robust (L1 + SSIM), but a terminal
            # convergence pass should measure and reduce the pixel error that drives the error
            # maps.  A tensor weight map supplied by the caller gives the inside boundary band
            # 5x weight while containment makes the unit-weight outside pixels identically zero.
            pixel_loss="l2",
            ssim_weight=0.0,
            loss_weighting="tensor",
            loss_weight_beta=4.0,
            lr_means=5e-3,
            lr_scales=3e-3,
            lr_rot=1e-3,
            lr_color=3e-3,
            lr_opacity=1e-3,
            lr_schedule="cosine",
            refine_site="none",
            refine_primitive="duplicate",
            refine_nms="off",
            # Cleanup has already run throughout all three growth stages. The terminal settle is
            # deliberately fixed-topology so its plateau measures optimization convergence rather
            # than another relocation/prune shock.
            prune_every=None,
            relocate_every=None,
            relocate_count=0,
        )
    else:
        raise ValueError(f"unknown stage mode {stage.mode!r}")
    return FitConfig(**kwargs)


def _find_source(bridge, capture_root: Path, frame_name: str, view_id: str):
    sources = [
        source
        for frame in bridge.CONVERTER.discover_source_frames(capture_root.resolve())
        for source in frame.views
        if source.frame.name == frame_name and source.view_id == view_id.upper()
    ]
    if len(sources) != 1:
        raise RuntimeError(
            f"expected exactly one {frame_name}/{view_id} below {capture_root}, found {len(sources)}"
        )
    if sources[0].mask_path is None:
        raise RuntimeError("selected Janelle source has no authoritative mask")
    return sources[0]


def _initial_field(target, mask, source_seed: int, device, margin: float):
    from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
    from structsplat.fit import _MaskConstraint, _boundary_tangent_add
    from structsplat.init import build_masked_field

    image_np = target.detach().cpu().numpy().astype(np.float32, copy=False)
    mask_np = mask.detach().cpu().numpy()
    init_cfg = InitConfig(
        strategy="quadtree_wse",
        num_gaussians=5_000,
        seed=int(source_seed),
        sampling_mode="wse",
        wse_progressive_order=True,
        init_scale_mult=0.35,
        scale_cap_mode="none",
        background_fraction=0.0,
        background_grid=0,
    )
    tensor_cfg = StructureTensorConfig()
    pool = build_masked_field(
        image_np,
        mask_np,
        init_cfg,
        tensor_cfg,
        device=str(device),
        sigma_cutoff=3.0,
        mask_margin=margin,
        contain=False,
    )
    field = pool.subset(slice(0, 4_500))
    cfg = FitConfig(
        iters=1,
        renderer="cuda_tiled",
        render_chunk=512,
        loss_weighting="mask",
        mask_contain=True,
        mask_margin=margin,
        mask_cap_mode="anisotropic",
        mask_cap_refresh_every=100,
        mask_boundary_add_every=1,
        mask_boundary_add_count=500,
        mask_boundary_add_band=4.0,
        mask_boundary_add_spacing=3.0,
        support_fade=True,
        max_gaussians=5_000,
        split_color_init="target",
        compute_lpips=False,
    )
    constraint = _MaskConstraint.from_mask(
        mask_np,
        device,
        target.dtype,
        cfg.sigma_cutoff,
        margin,
        aa_dilation=cfg.aa_dilation,
        cap_mode="anisotropic",
    )
    constraint.apply(field, cfg, refresh=True)
    with __import__("torch").no_grad():
        rendered = _render_field(field, cfg, mask.shape[0], mask.shape[1])
        field, boundary_added = _boundary_tangent_add(field, target, rendered, cfg, constraint)
        constraint.apply(field, cfg, refresh=True)
    if field.n != 5_000 or boundary_added != 500:
        raise RuntimeError(
            f"initial boundary allocation produced n={field.n}, boundary_added={boundary_added}; expected 5000/500"
        )
    return field, constraint, init_cfg, tensor_cfg, cfg, boundary_added


def _draw_initial_boundary_sites(path: Path, target, field, boundary_count: int, width: int) -> None:
    preview = _resize_hwc(target, width)
    image = _to_image(preview)
    draw = ImageDraw.Draw(image)
    scale = image.width / target.shape[1]
    points = field.means.detach().cpu().numpy()[-boundary_count:]
    radius = max(1.0, 1.5 * scale)
    for x, y in points:
        px, py = float(x) * scale, float(y) * scale
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(0, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=94, subsampling=0)


def _save_stage_output(path: Path, output: dict[str, Any], cfg, spec: StageSpec) -> None:
    record = {
        "spec": asdict(spec),
        "fit_config": asdict(cfg),
        "iterations_run": int(output["iterations_run"]),
        "stopped_early": bool(output["stopped_early"]),
        "stopped_at": output["stopped_at"],
        "n_gaussians": int(output["n_gaussians"]),
        "final_psnr": float(output["psnr"]),
        "terminal_psnr": float(output["trajectory_terminal_psnr"]),
        "selected_iter": int(output["selected_iter"]),
        "selected_from_checkpoint": bool(output["selected_from_checkpoint"]),
        "history": output["history"],
        "checkpoint_history": output["checkpoint_history"],
    }
    _atomic_json(path, record)


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from structsplat import metrics as M
    from structsplat.fit import _MaskConstraint, fit

    run_started = time.perf_counter()
    realtime_root = args.realtime_root.resolve()
    bridge = _load_bridge(realtime_root)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    executed_source_snapshots = _snapshot_executed_sources(out)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this native-resolution complete run requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    source = _find_source(bridge, args.capture_root.resolve(), args.frame, args.view_id)
    prepare_started = time.perf_counter()
    prepared = bridge.CONVERTER.prepare_source_view(source)
    if prepared.alpha_crop is None:
        raise RuntimeError("prepared source lost its mask")
    mask_cpu = prepared.alpha_crop.bool().contiguous()
    mask = mask_cpu.to(device)
    target = (prepared.rgb * mask_cpu[..., None].to(prepared.rgb)).to(device).contiguous()
    height, width = mask.shape
    outside = (~mask).to(target).view(1, 1, height, width)
    near_outside = F.max_pool2d(outside, kernel_size=5, stride=1, padding=2)[0, 0] > 0
    boundary = mask & near_outside
    prepare_seconds = time.perf_counter() - prepare_started

    margin = float(args.mask_margin)
    init_started = time.perf_counter()
    field, constraint, init_cfg, tensor_cfg, render_cfg, boundary_added = _initial_field(
        target, mask, source.seed, device, margin
    )
    init_seconds = time.perf_counter() - init_started
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    field.save(str(checkpoints / "00_initial_5000.npz"))
    _draw_initial_boundary_sites(
        out / "images/initial_boundary_sites.jpg",
        target * mask[..., None],
        field,
        boundary_added,
        args.preview_width,
    )

    writer = ProgressWriter(out, target, mask, boundary, render_cfg, args.preview_width)
    native_target_path = out / "images/native/target_foreground.png"
    native_target_path.parent.mkdir(parents=True, exist_ok=True)
    _to_image(target * mask[..., None].to(target)).save(native_target_path)
    initial_record = writer.snapshot(
        field, stage="initialization", global_iter=0, event="5000_with_boundary", native=False
    )
    run_record: dict[str, Any] = {
        "schema": "structsplat.janelle.complete_refinement.v1",
        "source": bridge._source_record(source),
        "fit_window": list(prepared.fit_window),
        "fit_size": [width, height],
        "foreground_pixels": int(mask.sum()),
        "boundary_pixels": int(boundary.sum()),
        "initialization": {
            "general_quadtree_wse": 4_500,
            "boundary_residual_tangent": boundary_added,
            "total": int(field.n),
            "init_config": asdict(init_cfg),
            "structure_tensor": asdict(tensor_cfg),
            "mask_margin": margin,
            "initial_metrics": initial_record,
        },
        "stages": [],
        "merges": [],
        "convergence": {},
        "repository": {
            **_git_record(),
            "executed_source_snapshots": executed_source_snapshots,
        },
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
        "timing": {
            "prepare_seconds": prepare_seconds,
            "initialization_seconds": init_seconds,
        },
    }
    _atomic_json(out / "run_config.json", run_record)

    global_iter = 0
    stage_index = 0
    for spec in GROWTH_STAGES:
        stage_index += 1
        stage_start_n = int(field.n)
        cfg = _fit_config(spec, margin=margin)
        print(
            f"\n=== stage {stage_index}: {spec.name} mode={spec.mode} "
            f"n={field.n} cap={spec.max_gaussians} ===",
            flush=True,
        )
        offset = global_iter

        def observer(observed_field, local_iter: int, loss: float, *, _offset=offset, _cfg=cfg, _name=spec.name):
            writer.cfg = _cfg
            writer.snapshot(
                observed_field,
                stage=_name,
                global_iter=_offset + int(local_iter),
                event="fit",
                loss=loss,
            )

        stage_started = time.perf_counter()
        output = fit(
            field,
            target,
            cfg,
            mask=mask_cpu.numpy(),
            verbose=True,
            iteration_observer=observer,
            observer_every=args.snapshot_every,
        )
        stage_seconds = time.perf_counter() - stage_started
        field = output["field"]
        constraint = _MaskConstraint.from_mask(
            mask_cpu.numpy(),
            device,
            target.dtype,
            cfg.sigma_cutoff,
            margin,
            aa_dilation=cfg.aa_dilation,
            cap_mode="anisotropic",
        )
        constraint.apply(field, cfg, refresh=True)
        global_iter += int(output["iterations_run"])
        writer.cfg = cfg
        stage_end = writer.snapshot(
            field,
            stage=spec.name,
            global_iter=global_iter,
            event="stage_end",
        )
        field.save(str(checkpoints / f"{stage_index:02d}_{spec.name}.npz"))
        _save_stage_output(
            checkpoints / f"{stage_index:02d}_{spec.name}.json", output, cfg, spec
        )
        stage_record = {
            "index": stage_index,
            "spec": asdict(spec),
            "fit_config": asdict(cfg),
            "iterations_run": int(output["iterations_run"]),
            "stopped_early": bool(output["stopped_early"]),
            "selected_iter": int(output["selected_iter"]),
            "seconds": stage_seconds,
            "end_metrics": stage_end,
            "split_events": output["history"]["split_events"],
            "boundary_add_events": output["history"]["boundary_add_events"],
            "inferred_pruned_rows_from_count_identity": int(
                stage_start_n
                + sum(event.get("added", 0) for event in output["history"]["split_events"])
                + sum(
                    event.get("added", 0)
                    for event in output["history"]["boundary_add_events"]
                )
                - int(field.n)
            ),
            "logged_count_decreases": int(
                sum(
                    1
                    for before, after in zip(
                        output["history"]["n_gaussians"],
                        output["history"]["n_gaussians"][1:],
                    )
                    if after < before
                )
            ),
            "relocate_events": output["history"]["relocate_events"],
        }
        run_record["stages"].append(stage_record)
        _atomic_json(out / "run_config.json", run_record)

        if spec.merge_after > 0:
            before = writer.snapshot(
                field,
                stage=spec.name,
                global_iter=global_iter,
                event="pre_merge",
            )
            with torch.no_grad():
                pre_merge_render = _render_field(field, cfg, height, width)
            merged, merge_stats = merge_redundant(
                field,
                spec.merge_after,
                constraint,
                cfg,
                target,
                pre_merge_render,
                mask,
            )
            field = merged
            after = writer.snapshot(
                field,
                stage=spec.name,
                global_iter=global_iter,
                event="post_merge",
            )
            merge_record = {
                "after_stage": spec.name,
                **merge_stats,
                "metrics_before": before,
                "metrics_after": after,
            }
            run_record["merges"].append(merge_record)
            field.save(str(checkpoints / f"{stage_index:02d}_{spec.name}_merged.npz"))
            _atomic_json(
                checkpoints / f"{stage_index:02d}_{spec.name}_merge.json", merge_record
            )
            _atomic_json(out / "run_config.json", run_record)

    # Fixed-count recovery rounds. Continue only while either foreground or boundary quality still
    # moves materially from one completed settle round to the next.
    settle_records: list[dict[str, Any]] = []
    previous = writer.records[-1]
    converged = False
    for settle_round in range(1, args.max_settle_rounds + 1):
        spec = StageSpec(
            name=f"settle_{settle_round}",
            mode="settle",
            max_gaussians=max(int(field.n), 5_000),
            iters=args.settle_iters,
            split_every=None,
            split_count=0,
            boundary_every=None,
            boundary_count=0,
            early_stop_min_iters=min(2_000, max(0, args.settle_iters - 1)),
        )
        cfg = _fit_config(spec, margin=margin)
        offset = global_iter
        incoming_field = field.detached()
        incoming_metrics = previous

        def settle_observer(observed_field, local_iter: int, loss: float, *, _offset=offset, _cfg=cfg, _name=spec.name):
            writer.cfg = _cfg
            writer.snapshot(
                observed_field,
                stage=_name,
                global_iter=_offset + int(local_iter),
                event="fit",
                loss=loss,
            )

        stage_started = time.perf_counter()
        output = fit(
            field,
            target,
            cfg,
            mask=mask_cpu.numpy(),
            loss_weight_map=boundary.detach().cpu().numpy(),
            verbose=True,
            iteration_observer=settle_observer,
            observer_every=args.snapshot_every,
        )
        seconds = time.perf_counter() - stage_started
        candidate_field = output["field"]
        constraint.apply(candidate_field, cfg, refresh=True)
        global_iter += int(output["iterations_run"])
        writer.cfg = cfg
        candidate = writer.snapshot(
            candidate_field,
            stage=spec.name,
            global_iter=global_iter,
            event="candidate_end",
        )
        # A terminal L2 pass can reduce mean absolute error while moving a few boundary outliers,
        # so raw boundary PSNR is allowed to move only inside its declared convergence tolerance.
        # Both foreground and boundary MAE must improve and foreground PSNR may not regress.  A
        # rejected candidate remains in progress.json so the decision is auditable.
        accepted = bool(
            candidate["foreground_psnr_raw_db"]
            >= incoming_metrics["foreground_psnr_raw_db"]
            and candidate["foreground_mae_clamped"]
            <= incoming_metrics["foreground_mae_clamped"]
            and candidate["boundary_mae_clamped"]
            <= incoming_metrics["boundary_mae_clamped"]
            and candidate["boundary_psnr_raw_db"]
            >= incoming_metrics["boundary_psnr_raw_db"] - args.convergence_boundary_db
        )
        if accepted:
            field = candidate_field
            end = candidate
        else:
            field = incoming_field
            constraint.apply(field, cfg, refresh=True)
            end = writer.snapshot(
                field,
                stage=spec.name,
                global_iter=global_iter,
                event="selected_incoming",
            )
        fg_gain = end["foreground_psnr_raw_db"] - previous["foreground_psnr_raw_db"]
        boundary_gain = end["boundary_psnr_raw_db"] - previous["boundary_psnr_raw_db"]
        settle_record = {
            "round": settle_round,
            "iterations_run": int(output["iterations_run"]),
            "stopped_early": bool(output["stopped_early"]),
            "seconds": seconds,
            "foreground_gain_db": fg_gain,
            "boundary_gain_db": boundary_gain,
            "end_metrics": end,
            "candidate_metrics": candidate,
            "candidate_accepted": accepted,
            "selection_rule": (
                "foreground PSNR nonworse; foreground and boundary MAE nonworse; "
                "boundary PSNR regression no larger than convergence-boundary-db"
            ),
            "relocate_events": output["history"]["relocate_events"],
        }
        settle_records.append(settle_record)
        field.save(str(checkpoints / f"{stage_index + settle_round:02d}_{spec.name}.npz"))
        _save_stage_output(
            checkpoints / f"{stage_index + settle_round:02d}_{spec.name}.json",
            output,
            cfg,
            spec,
        )
        if (
            abs(fg_gain) < args.convergence_fg_db
            and abs(boundary_gain) < args.convergence_boundary_db
            and output["stopped_early"]
        ):
            converged = True
            break
        previous = end

    final_cfg = cfg
    final_record = writer.snapshot(
        field,
        stage="final_full_field",
        global_iter=global_iter,
        event="terminal",
        native=True,
    )
    field.save(str(out / "C0001_full_refined.npz"))

    with torch.no_grad():
        final_render = _render_field(field, final_cfg, height, width)
        final_ms_ssim = float(M.ms_ssim(final_render.clamp(0.0, 1.0), target))
        final_ssim = float(M.ssim(final_render.clamp(0.0, 1.0), target))

    # Derive the strict byte-capped transport archive. This is intentionally separate from the
    # full-field result; activity backoff can remove boundary-important rows.
    activity = (
        bridge.CONVERTER._component_activity(
            field,
            mask,
            height=height,
            width=width,
        )
        .detach()
        .cpu()
        .numpy()
    )
    binding = bridge.CONVERTER._implementation_binding()
    contract = {
        "schema": run_record["schema"],
        "source": run_record["source"],
        "initialization": run_record["initialization"],
        "growth_stages": [asdict(spec) for spec in GROWTH_STAGES],
        "settle": settle_records,
        "repository": run_record["repository"],
    }
    fit_digest = hashlib.sha256(
        json.dumps(_jsonable(contract), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    observation = bridge.field_to_observation(
        field,
        canvas_size=(prepared.camera.height, prepared.camera.width),
        fit_window=prepared.fit_window,
        blend_mode="normalized",
        sigma_cutoff=final_cfg.sigma_cutoff,
        support_fade_alpha=1.0,
        aa_dilation=final_cfg.aa_dilation,
        view_id=source.view_id,
        n_init=5_000,
        producer_version=binding["structsplat_version"],
        producer_source_digest=binding["structsplat_source_sha256"],
        fit_config_digest=fit_digest,
    ).to("cpu")
    archive_path = out / "gaussians2d/C0001.rtgsv"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    strict_view = bridge.CONVERTER._save_with_backoff(
        archive_path,
        prepared,
        observation,
        np.asarray(activity, dtype=np.float64),
    )
    archive_render = bridge._render_selected(strict_view.observation, device, "cuda_tiled")
    archive_metrics = _metric_record(archive_render, target, mask, boundary)
    archive_metrics.update(
        {
            "bytes": int(strict_view.bytes),
            "sha256": bridge.file_sha256(archive_path),
            "n_gaussians": int(strict_view.observation.n),
            "rows_removed_from_full_field": int(field.n - strict_view.observation.n),
        }
    )
    archive_cfg = replace(final_cfg, renderer="cuda_tiled")
    writer.cfg = archive_cfg
    archive_field = bridge._selected_field(strict_view.observation, device)
    writer.snapshot(
        archive_field,
        stage="byte_capped_archive",
        global_iter=global_iter,
        event="terminal",
        native=True,
    )

    run_record["convergence"] = {
        "criterion": (
            "after an early-stopped settle round, abs foreground gain < "
            f"{args.convergence_fg_db} dB and abs boundary gain < "
            f"{args.convergence_boundary_db} dB"
        ),
        "converged": converged,
        "settle_rounds": settle_records,
        "global_iterations": global_iter,
    }
    run_record["final_full_field"] = {
        "n_gaussians": int(field.n),
        "metrics": final_record,
        "ssim_clamped_matted_crop": final_ssim,
        "ms_ssim_clamped_matted_crop": final_ms_ssim,
        "path": "C0001_full_refined.npz",
        "sha256": bridge.file_sha256(out / "C0001_full_refined.npz"),
    }
    run_record["byte_capped_archive"] = {
        "path": str(archive_path.relative_to(out)),
        "metrics": archive_metrics,
        "selection": "foreground-activity-ranked deterministic backoff",
        "warning": "archive selection is not boundary-error-aware; compare with the full native field",
    }
    run_record["timing"].update(
        {
            "optimization_and_output_seconds": time.perf_counter() - writer.started,
            "total_seconds": time.perf_counter() - run_started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    )
    _atomic_json(out / "run_config.json", run_record)
    _atomic_json(out / "summary.json", {
        "source": f"{args.frame}/{args.view_id.upper()}",
        "initial": initial_record,
        "full_field": run_record["final_full_field"],
        "archive": run_record["byte_capped_archive"],
        "convergence": run_record["convergence"],
        "operators": {
            "boundary_initialized": boundary_added,
            "high_error_birth": True,
            "moment_preserving_split": True,
            "fit_time_prune": True,
            "residual_relocation_teleportation": True,
            "between_stage_count_neutral_merge_teleport": True,
        },
    })
    print(
        f"\ncomplete: full field n={field.n}, fg={final_record['foreground_psnr_raw_db']:.3f}dB, "
        f"boundary={final_record['boundary_psnr_raw_db']:.3f}dB; "
        f"archive n={strict_view.observation.n}, bytes={strict_view.bytes}; out={out}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--realtime-root", type=Path, default=DEFAULT_REALTIME_ROOT)
    parser.add_argument("--frame", default="frame_00008")
    parser.add_argument("--view-id", default="C0001")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mask-margin", type=float, default=0.75)
    parser.add_argument("--preview-width", type=int, default=1200)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument("--settle-iters", type=int, default=6_000)
    parser.add_argument("--max-settle-rounds", type=int, default=3)
    parser.add_argument("--convergence-fg-db", type=float, default=0.01)
    parser.add_argument("--convergence-boundary-db", type=float, default=0.03)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mask_margin < 0.72:
        raise ValueError("--mask-margin must stay above the ~0.71 px containment floor")
    if args.preview_width <= 0 or args.snapshot_every <= 0:
        raise ValueError("preview width and snapshot cadence must be positive")
    if args.settle_iters <= 0 or args.max_settle_rounds <= 0:
        raise ValueError("settle horizon and round count must be positive")
    _prepend_source_roots(args.realtime_root.resolve())
    run(args)


if __name__ == "__main__":
    main()
