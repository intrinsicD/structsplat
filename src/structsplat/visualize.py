"""Deterministic publication diagnostics for the StructSplat representation.

This module visualizes the *existing* initialization and normalized renderer.  It does not fit a
field, run an ablation, or turn an initialization screenshot into quality evidence (DOCS-002).
The raw diagnostic arrays and the display transforms are saved alongside every figure bundle.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import density as density_ops
from . import render as render_ops
from . import structure_tensor as tensor_ops
from .config import InitConfig, StructureTensorConfig
from .gaussians import GaussianField
from .init import STRATEGIES, build_field


SCHEMA_VERSION = 1
LABEL_COLORS: dict[int, tuple[int, int, int]] = {
    0: (105, 105, 105),  # flat: neutral gray
    1: (230, 159, 0),  # edge: orange
    2: (86, 180, 233),  # corner: sky blue
}
LABEL_NAMES = {0: "flat", 1: "edge", 2: "corner"}
_IMPLEMENTATION_MODULES = (
    "config.py",
    "density.py",
    "gaussians.py",
    "init.py",
    "render.py",
    "sampling.py",
    "structure_tensor.py",
    "visualize.py",
)
_HEATMAP_ANCHORS = (
    np.asarray(
        [
            (38, 18, 74),
            (49, 74, 142),
            (34, 132, 141),
            (94, 190, 104),
            (250, 231, 62),
        ],
        dtype=np.float64,
    )
    / 255.0
)


@dataclass(frozen=True)
class FigureConfig:
    """Resolved, serializable controls for one explanatory figure bundle."""

    max_side: int | None = 256
    crop: tuple[int, int, int, int] | None = None  # x, y, width, height before resize
    strategy: str = "aniso_onedge"
    num_gaussians: int = 384
    seed: int = 0
    candidate_oversample: float = 4.0
    density_mode: str = "structure"
    density_base: float = 0.05
    density_power: float = 1.0
    max_axis_ratio: float = 6.0
    coherence_power: float = 1.0
    init_scale_mult: float = 1.0
    gradient_operator: str = "central"
    color_space: str = "luma"
    grad_sigma: float = 1.0
    tensor_sigma: float = 2.0
    flat_frac: float = 0.02
    corner_frac: float = 0.15
    sigma_cutoff: float = 3.0
    glyph_spacing: int = 18
    glyph_length: float = 8.0
    glyph_min_coherence: float = 0.10
    ellipse_limit: int = 220
    metric_ellipse_limit: int = 72
    montage_columns: int = 4

    def __post_init__(self) -> None:
        if self.max_side is not None and self.max_side <= 0:
            raise ValueError("max_side must be positive or None")
        if self.crop is not None and (
            len(self.crop) != 4
            or any(int(v) < 0 for v in self.crop[:2])
            or any(int(v) <= 0 for v in self.crop[2:])
        ):
            raise ValueError("crop must be (x, y, width, height) with positive extent")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}; expected one of {STRATEGIES}")
        if self.strategy == "feedforward":
            raise ValueError("feedforward is not supported by the explanatory figure generator")
        if self.num_gaussians <= 0:
            raise ValueError("num_gaussians must be positive")
        if self.candidate_oversample < 1.0:
            raise ValueError("candidate_oversample must be at least 1")
        if self.max_axis_ratio < 1.0:
            raise ValueError("max_axis_ratio must be at least 1")
        if self.sigma_cutoff <= 0.0:
            raise ValueError("sigma_cutoff must be positive")
        if self.glyph_spacing <= 0 or self.glyph_length <= 0.0:
            raise ValueError("glyph spacing and length must be positive")
        if not (0.0 <= self.glyph_min_coherence <= 1.0):
            raise ValueError("glyph_min_coherence must be in [0, 1]")
        if self.ellipse_limit <= 0 or self.metric_ellipse_limit <= 0:
            raise ValueError("ellipse limits must be positive")
        if self.montage_columns <= 0:
            raise ValueError("montage_columns must be positive")


@dataclass(frozen=True)
class ResponsibilityDiagnostics:
    """Per-pixel diagnostics for normalized Gaussian responsibilities."""

    denominator: np.ndarray
    responsibility_sum: np.ndarray
    effective_count: np.ndarray
    entropy: np.ndarray
    active_count: np.ndarray
    dominant_owner: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_png(path: Path, rgb: np.ndarray | Image.Image) -> None:
    image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb, mode="RGB")
    image.convert("RGB").save(path, format="PNG", compress_level=9, optimize=False)


def _deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ with stable member order and timestamps."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            stream = io.BytesIO()
            np.lib.format.write_array(stream, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, stream.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _git_provenance(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    diff = run("diff", "--binary", "HEAD", "--", ".")
    return {
        "commit": commit,
        "tracked_diff_sha256": (
            None if diff is None else hashlib.sha256(diff.encode("utf-8")).hexdigest()
        ),
        "tracked_tree_dirty": bool(diff),
    }


def _implementation_provenance(root: Path) -> dict[str, dict[str, Any]]:
    """Hash the production sources used by the explanatory pipeline.

    These hashes deliberately complement Git provenance: ``git diff`` does not include untracked
    files, while a figure bundle may be generated before its implementation has been committed.
    """
    package_dir = Path(__file__).resolve().parent
    provenance: dict[str, dict[str, Any]] = {}
    for name in _IMPLEMENTATION_MODULES:
        path = package_dir / name
        if not path.is_file():
            continue
        provenance[_relative_source_path(path, root)] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return provenance


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def load_figure_image(
    path: str | Path,
    max_side: int | None = 256,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load, optional-crop, and optional-resize an RGB image to float32 ``[0,1]``."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as opened:
        oriented = ImageOps.exif_transpose(opened).convert("RGB")
        source_size = oriented.size
        if crop is not None:
            x, y, width, height = (int(v) for v in crop)
            if x + width > oriented.width or y + height > oriented.height:
                raise ValueError(f"crop {crop} exceeds oriented source size {oriented.size}")
            oriented = oriented.crop((x, y, x + width, y + height))
        cropped_size = oriented.size
        if max_side is not None and max(oriented.size) > max_side:
            scale = float(max_side) / max(oriented.size)
            resized = (
                max(1, int(round(oriented.width * scale))),
                max(1, int(round(oriented.height * scale))),
            )
            oriented = oriented.resize(resized, Image.Resampling.LANCZOS)
        array = np.asarray(oriented, dtype=np.float32) / np.float32(255.0)
    return array, {
        "source_path": str(source),
        "source_sha256": _sha256_file(source),
        "source_oriented_size": list(source_size),
        "crop_xywh": None if crop is None else [int(v) for v in crop],
        "cropped_size": list(cropped_size),
        "processed_size": [int(array.shape[1]), int(array.shape[0])],
        "resize_filter": "Pillow LANCZOS" if cropped_size != oriented.size else "none",
    }


def _display_normalize(
    values: np.ndarray,
    *,
    percentile: float = 99.0,
    log_gain: float | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    data = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    positive = data[data > 0.0]
    reference = float(np.percentile(positive, percentile)) if positive.size else 1.0
    reference = max(reference, np.finfo(np.float64).eps)
    normalized = np.clip(data / reference, 0.0, 1.0)
    transform = "linear"
    if log_gain is not None:
        normalized = np.log1p(float(log_gain) * normalized) / math.log1p(float(log_gain))
        transform = f"log1p(gain={float(log_gain):g})"
    return normalized.astype(np.float32), {
        "transform": transform,
        "reference_percentile": float(percentile),
        "reference_value": reference,
        "display_clamp": "[0,1] after reference scaling",
    }


def scalar_heatmap(normalized: np.ndarray) -> np.ndarray:
    """Map a normalized scalar field to a fixed, monotone, perceptual-style palette."""
    value = np.clip(np.asarray(normalized, dtype=np.float64), 0.0, 1.0)
    scaled = value * (_HEATMAP_ANCHORS.shape[0] - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, _HEATMAP_ANCHORS.shape[0] - 1)
    fraction = scaled - lower
    rgb = (
        _HEATMAP_ANCHORS[lower] * (1.0 - fraction[..., None])
        + _HEATMAP_ANCHORS[upper] * fraction[..., None]
    )
    return np.round(255.0 * rgb).astype(np.uint8)


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, color in LABEL_COLORS.items():
        rgb[labels == label] = color
    unknown = ~np.isin(labels, list(LABEL_COLORS))
    rgb[unknown] = (255, 0, 255)
    return rgb


def tensor_glyphs(
    tensor: tensor_ops.StructureTensor,
    spacing: int,
    min_coherence: float = 0.10,
    flat_frac: float = 0.02,
) -> np.ndarray:
    """Select one high-energy tangent glyph per regular cell.

    Rows are ``(x, y, tangent_angle, coherence, energy)`` in processed-image pixel coordinates.
    The local energy maximum avoids placing a glyph on a flat cell center beside a thin feature.
    """
    height, width = tensor.energy.shape
    threshold = tensor_ops.flat_threshold(tensor.energy, flat_frac, tensor.energy_ref)
    rows: list[tuple[float, float, float, float, float]] = []
    for y0 in range(0, height, spacing):
        y1 = min(y0 + spacing, height)
        for x0 in range(0, width, spacing):
            x1 = min(x0 + spacing, width)
            local = tensor.energy[y0:y1, x0:x1]
            if local.size == 0:
                continue
            offset = int(np.argmax(local))
            dy, dx = np.unravel_index(offset, local.shape)
            x, y = x0 + int(dx), y0 + int(dy)
            energy = float(tensor.energy[y, x])
            coherence = float(tensor.coherence[y, x])
            if energy < threshold or coherence < min_coherence or tensor.label[y, x] == 0:
                continue
            rows.append((x, y, float(tensor.along_edge_angle[y, x]), coherence, energy))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def _draw_tensor_glyphs(
    image: np.ndarray,
    glyphs: np.ndarray,
    base_length: float,
) -> Image.Image:
    scale = 3
    base = Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8), mode="RGB")
    canvas = base.resize((base.width * scale, base.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas, mode="RGB")
    for x, y, tangent, coherence, _ in glyphs:
        length = base_length * (0.45 + 0.55 * float(coherence))
        tx, ty = math.cos(tangent), math.sin(tangent)
        nx, ny = -ty, tx
        tangent_segment = (
            ((x - length * tx) * scale, (y - length * ty) * scale),
            ((x + length * tx) * scale, (y + length * ty) * scale),
        )
        normal_length = 0.32 * length
        normal_segment = (
            ((x - normal_length * nx) * scale, (y - normal_length * ny) * scale),
            ((x + normal_length * nx) * scale, (y + normal_length * ny) * scale),
        )
        draw.line(tangent_segment, fill=(0, 220, 255), width=2 * scale)
        draw.line(normal_segment, fill=(255, 112, 67), width=scale)
        radius = 1.2 * scale
        draw.ellipse(
            (x * scale - radius, y * scale - radius, x * scale + radius, y * scale + radius),
            fill=(255, 255, 255),
        )
    return canvas.resize(base.size, Image.Resampling.LANCZOS)


def _nearest(map2d: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = map2d.shape[:2]
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
    return map2d[y, x]


def _even_subset(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, count - 1, limit)).astype(np.int64))


def _ellipse_polygon(
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    scale: int,
    samples: int = 40,
) -> list[tuple[float, float]]:
    phase = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    local = np.stack([axes[0] * np.cos(phase), axes[1] * np.sin(phase)], axis=1)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    points = local @ rotation.T + center
    return [(float(x * scale), float(y * scale)) for x, y in points]


def _sampling_panel(
    density_rgb: np.ndarray,
    field: GaussianField,
    tensor: tensor_ops.StructureTensor,
    cfg: FigureConfig,
) -> Image.Image:
    scale = 3
    base = Image.fromarray(density_rgb, mode="RGB")
    canvas = base.resize((base.width * scale, base.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas, mode="RGBA")
    means = field.means.detach().cpu().numpy()
    indices = _even_subset(len(means), cfg.metric_ellipse_limit)
    if len(indices):
        coherence = np.clip(_nearest(tensor.coherence, means[indices]), 0.0, 1.0)
        ratio = 1.0 + (cfg.max_axis_ratio - 1.0) * coherence**cfg.coherence_power
        tangent = _nearest(tensor.along_edge_angle, means[indices])
        base_radius = max(1.5, 0.23 * cfg.glyph_spacing)
        for point, local_ratio, angle in zip(means[indices], ratio, tangent):
            axes = base_radius * np.asarray(
                [math.sqrt(float(local_ratio)), 1.0 / math.sqrt(float(local_ratio))]
            )
            polygon = _ellipse_polygon(point, axes, float(angle), scale)
            draw.line(polygon + [polygon[0]], fill=(255, 255, 255, 205), width=scale)
    radius = max(1, scale)
    for x, y in means:
        draw.ellipse(
            (x * scale - radius, y * scale - radius, x * scale + radius, y * scale + radius),
            fill=(255, 64, 129, 235),
        )
    return canvas.resize(base.size, Image.Resampling.LANCZOS)


def _gaussian_panel(image: np.ndarray, field: GaussianField, limit: int) -> Image.Image:
    scale = 3
    dimmed = np.clip(image * 0.38 + 0.10, 0.0, 1.0)
    base = Image.fromarray(np.round(dimmed * 255).astype(np.uint8), mode="RGB")
    canvas = base.resize((base.width * scale, base.height * scale), Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, mode="RGBA")
    means = field.means.detach().cpu().numpy()
    scales = field.effective_scales().detach().cpu().numpy()
    angles = field.rotations.detach().cpu().numpy()
    colors = np.clip(field.colors.detach().cpu().numpy(), 0.0, 1.0)
    for index in _even_subset(len(means), limit):
        polygon = _ellipse_polygon(means[index], scales[index], float(angles[index]), scale)
        color = tuple(int(round(255 * channel)) for channel in colors[index])
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        outline = (255, 255, 255, 235) if luminance < 115 else (18, 18, 18, 235)
        draw.polygon(polygon, fill=(*color, 82))
        draw.line(polygon + [polygon[0]], fill=outline, width=scale)
        x, y = means[index] * scale
        radius = 1.0 * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=outline)
    return (
        Image.alpha_composite(canvas.convert("RGBA"), overlay)
        .convert("RGB")
        .resize(base.size, Image.Resampling.LANCZOS)
    )


def gaussian_overlay(
    image: np.ndarray,
    field: GaussianField,
    limit: int = 500,
) -> Image.Image:
    """Overlay fitted Gaussian one-sigma ellipses on an RGB image.

    This is a read-only display helper for decoded/fitted fields.  It complements the
    initialization-only method bundle without recomputing or claiming source-derived tensor
    features for a field that no longer has access to its source image.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H,W,3), got {image.shape}")
    if limit <= 0:
        raise ValueError("limit must be positive")
    return _gaussian_panel(image, field, limit)


def responsibility_diagnostics(
    field: GaussianField,
    height: int,
    width: int,
    *,
    sigma_cutoff: float = 3.0,
    support_fade: bool = False,
    chunk: int = 512,
) -> ResponsibilityDiagnostics:
    """Compute normalized-renderer ownership diagnostics using its exact support helpers.

    The renderer evaluates every Gaussian on its clipped axis-aligned support rectangle.  This
    routine calls the same tile-bound, coordinate, and support-weight helpers, including opacity
    and the optional compact-support fade.  It changes no field state and is intended for audit
    figures rather than gradient computation.
    """
    with torch.no_grad():
        means = field.means.detach()
        conics = field.conics().detach()
        radii = field.radii(sigma_cutoff).detach()
        opacities = field.opacity_values()
        if opacities is not None:
            opacities = opacities.detach()
        device, dtype = means.device, means.dtype
        pixel_count = int(height * width)
        denominator = torch.zeros(pixel_count, device=device, dtype=dtype)
        squared = torch.zeros_like(denominator)
        weight_log_weight = torch.zeros_like(denominator)
        active = torch.zeros(pixel_count, device=device, dtype=torch.int64)

        x0, y0, tile_width, tile_size = render_ops._tile_bounds(means, radii, height, width)
        budget = render_ops._element_budget(chunk)

        def contributions() -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            for start, end in render_ops._flat_tile_slices(tile_size, budget):
                gid, px, py = render_ops._tile_coords(
                    x0, y0, tile_width, tile_size, start, end, device
                )
                dx = px.to(dtype) - means[gid, 0]
                dy = py.to(dtype) - means[gid, 1]
                a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
                quadratic = a * dx.square() + 2.0 * b * dx * dy + c * dy.square()
                weight = render_ops._support_weight(quadratic, sigma_cutoff, support_fade, None)
                if opacities is not None:
                    weight = weight * opacities[gid]
                yield gid, py * width + px, weight

        maximum = torch.full((pixel_count,), -torch.inf, device=device, dtype=dtype)
        for _, flat, weight in contributions():
            denominator.index_add_(0, flat, weight)
            squared.index_add_(0, flat, weight.square())
            positive = weight > 0
            safe_log = torch.where(positive, torch.log(weight.clamp_min(1e-30)), 0.0)
            weight_log_weight.index_add_(0, flat, weight * safe_log)
            active.index_add_(0, flat, positive.to(torch.int64))
            maximum.scatter_reduce_(0, flat, weight, reduce="amax", include_self=True)

        owner = torch.full((pixel_count,), field.n, device=device, dtype=torch.int64)
        for gid, flat, weight in contributions():
            winner = torch.where(weight == maximum[flat], gid, field.n)
            owner.scatter_reduce_(0, flat, winner, reduce="amin", include_self=True)

        covered = denominator > 0
        responsibility_sum = torch.zeros_like(denominator)
        responsibility_sum[covered] = 1.0
        effective = torch.zeros_like(denominator)
        effective[covered] = denominator[covered].square() / squared[covered].clamp_min(1e-30)
        entropy = torch.zeros_like(denominator)
        entropy[covered] = (
            torch.log(denominator[covered]) - weight_log_weight[covered] / denominator[covered]
        ).clamp_min(0.0)
        owner[~covered] = -1

        def image(array: torch.Tensor, *, integer: bool = False) -> np.ndarray:
            result = array.reshape(height, width).detach().cpu().numpy()
            return result.astype(np.int32 if integer else np.float32, copy=False)

        return ResponsibilityDiagnostics(
            denominator=image(denominator),
            responsibility_sum=image(responsibility_sum),
            effective_count=image(effective),
            entropy=image(entropy),
            active_count=image(active, integer=True),
            dominant_owner=image(owner, integer=True),
        )


def _owner_rgb(owner: np.ndarray) -> np.ndarray:
    owner = np.asarray(owner, dtype=np.int64)
    output = np.full((*owner.shape, 3), (15, 15, 20), dtype=np.uint8)
    for index in np.unique(owner[owner >= 0]):
        hue = (float(index) * 0.6180339887498948) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.64, 0.94)
        output[owner == index] = np.round(255 * np.asarray(rgb)).astype(np.uint8)
    return output


def _render_initial_field(
    field: GaussianField,
    height: int,
    width: int,
    sigma_cutoff: float,
) -> np.ndarray:
    with torch.no_grad():
        reconstruction = render_ops.render_field(
            field.means,
            field.conics(),
            field.colors,
            field.radii(sigma_cutoff),
            height,
            width,
            mode="normalized",
            opacities=field.opacity_values(),
            scales=field.effective_scales(),
            rotations=field.rotations,
            color_grads=field.color_grads,
            sigma_cutoff=sigma_cutoff,
        )
    return reconstruction.detach().cpu().numpy().astype(np.float32)


def _montage(
    panels: list[tuple[str, Path]],
    columns: int,
    disclaimer: str,
) -> Image.Image:
    images = [(title, Image.open(path).convert("RGB")) for title, path in panels]
    tile_width = max(image.width for _, image in images)
    tile_height = max(image.height for _, image in images)
    label_height, footer_height, gap = 24, 26, 6
    rows = int(math.ceil(len(images) / columns))
    width = columns * tile_width + (columns - 1) * gap
    height = rows * (tile_height + label_height) + (rows - 1) * gap + footer_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (title, image) in enumerate(images):
        row, column = divmod(index, columns)
        x = column * (tile_width + gap)
        y = row * (tile_height + label_height + gap)
        canvas.paste(image, (x, y))
        draw.rectangle(
            (x, y + tile_height, x + tile_width, y + tile_height + label_height),
            fill=(245, 245, 245),
        )
        draw.text((x + 5, y + tile_height + 6), title, fill=(20, 20, 20), font=font)
    footer_y = height - footer_height
    draw.rectangle((0, footer_y, width, height), fill=(35, 35, 40))
    draw.text((6, footer_y + 7), disclaimer, fill=(245, 245, 245), font=font)
    return canvas


def _method_overview_svg() -> str:
    """Return a deterministic vector overview of the encoder/decoder data flow."""
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="560" viewBox="0 0 1440 560" role="img" aria-labelledby="title desc">
  <title id="title">StructSplat encoder and decoder overview</title>
  <desc id="desc">The source-visible encoder computes a structure tensor, initializes and fits a Gaussian field, and writes an SSPL1 stream. The stream-only decoder reconstructs the field and applies the normalized renderer. Tensor maps are not transmitted.</desc>
  <defs>
    <style>
      .title { font-family: sans-serif; font-size: 25px; font-weight: bold; fill: #111827; }
      .lane { font-family: sans-serif; font-size: 16px; font-weight: bold; letter-spacing: 1px; fill: #334155; }
      .head { font-family: sans-serif; font-size: 17px; font-weight: bold; fill: #111827; }
      .body { font-family: sans-serif; font-size: 14px; fill: #334155; }
      .small { font-family: sans-serif; font-size: 13px; fill: #475569; }
      .box { stroke-width: 2; rx: 12; ry: 12; }
      .flow { font-family: sans-serif; font-size: 28px; font-weight: bold; fill: #334155; }
    </style>
  </defs>
  <rect width="1440" height="560" fill="#ffffff"/>
  <text x="40" y="38" class="title">StructSplat: source-derived initialization, transmitted Gaussian state</text>

  <rect x="28" y="62" width="1384" height="232" rx="18" fill="#f4f8ff" stroke="#bfdbfe"/>
  <text x="48" y="91" class="lane">ENCODER · SOURCE VISIBLE</text>

  <rect class="box" x="45" y="116" width="145" height="100" fill="#ffffff" stroke="#64748b"/>
  <text x="117" y="148" text-anchor="middle" class="head">Input image</text>
  <text x="117" y="174" text-anchor="middle" class="body">RGB target</text>
  <text x="117" y="196" text-anchor="middle" class="small">not transmitted</text>

  <text x="202" y="175" class="flow">→</text>
  <rect class="box" x="230" y="116" width="195" height="100" fill="#e0f2fe" stroke="#0284c7"/>
  <text x="327" y="143" text-anchor="middle" class="head">Structure tensor</text>
  <text x="327" y="169" text-anchor="middle" class="body">energy · coherence</text>
  <text x="327" y="191" text-anchor="middle" class="body">labels · tangent</text>

  <text x="437" y="175" class="flow">→</text>
  <rect class="box" x="465" y="108" width="225" height="116" fill="#e0f2fe" stroke="#0284c7"/>
  <text x="577" y="137" text-anchor="middle" class="head">Tensor/WSE initialization</text>
  <text x="577" y="163" text-anchor="middle" class="body">density + unit-area metric</text>
  <text x="577" y="185" text-anchor="middle" class="body">exact-N sites + tangent RS</text>
  <text x="577" y="207" text-anchor="middle" class="small">training-free starting field</text>

  <text x="702" y="175" class="flow">→</text>
  <rect class="box" x="730" y="116" width="170" height="100" fill="#f1f5f9" stroke="#64748b"/>
  <text x="815" y="147" text-anchor="middle" class="head">Fit field</text>
  <text x="815" y="173" text-anchor="middle" class="body">normalized render</text>
  <text x="815" y="195" text-anchor="middle" class="body">+ image loss</text>

  <text x="912" y="175" class="flow">→</text>
  <rect class="box" x="940" y="116" width="170" height="100" fill="#fef3c7" stroke="#d97706"/>
  <text x="1025" y="147" text-anchor="middle" class="head">Codec</text>
  <text x="1025" y="173" text-anchor="middle" class="body">quantize attributes</text>
  <text x="1025" y="195" text-anchor="middle" class="body">+ pack stream</text>

  <text x="1122" y="175" class="flow">→</text>
  <rect class="box" x="1150" y="116" width="205" height="100" fill="#ffedd5" stroke="#ea580c"/>
  <text x="1252" y="148" text-anchor="middle" class="head">SSPL1 bytes</text>
  <text x="1252" y="174" text-anchor="middle" class="body">decoder-complete state</text>
  <text x="1252" y="196" text-anchor="middle" class="small">actual rate = file bytes</text>

  <text x="48" y="260" class="small">Encoder-only tensor, density, and WSE analysis is discarded; the decoder receives no source-derived maps.</text>

  <rect x="28" y="316" width="1384" height="208" rx="18" fill="#fafafa" stroke="#d1d5db"/>
  <text x="48" y="346" class="lane">DECODER · STREAM ONLY</text>
  <polyline points="1252,216 1252,280 250,280 250,365" stroke="#ea580c" stroke-width="2.5" stroke-dasharray="7 5" fill="none"/>

  <rect class="box" x="145" y="378" width="210" height="92" fill="#ffedd5" stroke="#ea580c"/>
  <text x="250" y="411" text-anchor="middle" class="head">SSPL1 bytes</text>
  <text x="250" y="439" text-anchor="middle" class="body">cold-decode input</text>

  <text x="371" y="433" class="flow">→</text>
  <rect class="box" x="415" y="378" width="205" height="92" fill="#f1f5f9" stroke="#64748b"/>
  <text x="517" y="411" text-anchor="middle" class="head">Decoded RS field</text>
  <text x="517" y="439" text-anchor="middle" class="body">position · scale · angle · color</text>

  <text x="636" y="433" class="flow">→</text>
  <rect class="box" x="680" y="378" width="210" height="92" fill="#f1f5f9" stroke="#64748b"/>
  <text x="785" y="411" text-anchor="middle" class="head">Normalized renderer</text>
  <text x="785" y="439" text-anchor="middle" class="body">local clipped support</text>

  <text x="906" y="433" class="flow">→</text>
  <rect class="box" x="950" y="378" width="170" height="92" fill="#dcfce7" stroke="#16a34a"/>
  <text x="1035" y="411" text-anchor="middle" class="head">Decoded image</text>
  <text x="1035" y="439" text-anchor="middle" class="body">RGB reconstruction</text>

  <rect x="1155" y="368" width="225" height="112" rx="10" fill="#ffffff" stroke="#94a3b8"/>
  <text x="1267" y="397" text-anchor="middle" class="head">Forward equation</text>
  <text x="1267" y="426" text-anchor="middle" class="body">I-hat(p) = sum w_i c_i</text>
  <line x1="1192" y1="434" x2="1342" y2="434" stroke="#64748b"/>
  <text x="1267" y="456" text-anchor="middle" class="body">sum w_i + epsilon</text>

  <rect x="48" y="492" width="14" height="14" fill="#e0f2fe" stroke="#0284c7"/>
  <text x="70" y="504" class="small">source-only analysis</text>
  <rect x="215" y="492" width="14" height="14" fill="#ffedd5" stroke="#ea580c"/>
  <text x="237" y="504" class="small">transmitted state</text>
  <rect x="365" y="492" width="14" height="14" fill="#f1f5f9" stroke="#64748b"/>
  <text x="387" y="504" class="small">shared method</text>
</svg>
"""


def _relative_source_path(source: Path, root: Path) -> str:
    try:
        return str(source.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(source.resolve())


def generate_figure_bundle(
    image_path: str | Path,
    outdir: str | Path,
    cfg: FigureConfig | None = None,
) -> dict[str, Any]:
    """Generate individual panels, raw maps, a montage, and a provenance manifest."""
    cfg = cfg or FigureConfig()
    source = Path(image_path)
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()

    image, source_info = load_figure_image(source, cfg.max_side, cfg.crop)
    height, width = image.shape[:2]
    tensor_cfg = StructureTensorConfig(
        grad_sigma=cfg.grad_sigma,
        tensor_sigma=cfg.tensor_sigma,
        gradient_operator=cfg.gradient_operator,
        color_space=cfg.color_space,
        flat_frac=cfg.flat_frac,
        corner_frac=cfg.corner_frac,
    )
    init_cfg = InitConfig(
        strategy=cfg.strategy,
        num_gaussians=cfg.num_gaussians,
        candidate_oversample=cfg.candidate_oversample,
        density_base=cfg.density_base,
        density_power=cfg.density_power,
        density_mode=cfg.density_mode,
        max_axis_ratio=cfg.max_axis_ratio,
        coherence_power=cfg.coherence_power,
        init_scale_mult=cfg.init_scale_mult,
        sampling_mode="wse",
        seed=cfg.seed,
    )
    tensor = tensor_ops.compute(image, tensor_cfg)
    density = density_ops.density_from_tensor_and_image(image, tensor, init_cfg, tensor_cfg)
    field = build_field(image, init_cfg, tensor_cfg, density=density, tensor=tensor, device="cpu")
    reconstruction = _render_initial_field(field, height, width, cfg.sigma_cutoff)
    diagnostics = responsibility_diagnostics(field, height, width, sigma_cutoff=cfg.sigma_cutoff)

    input_rgb = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    energy_display, energy_scale = _display_normalize(tensor.energy, percentile=99.0, log_gain=9.0)
    density_display, density_scale = _display_normalize(density, percentile=99.0, log_gain=9.0)
    denominator_display, denominator_scale = _display_normalize(
        diagnostics.denominator, percentile=99.0, log_gain=9.0
    )
    effective_display, effective_scale = _display_normalize(
        diagnostics.effective_count, percentile=99.0
    )
    entropy_display, entropy_scale = _display_normalize(diagnostics.entropy, percentile=99.0)
    glyph_rows = tensor_glyphs(tensor, cfg.glyph_spacing, cfg.glyph_min_coherence, cfg.flat_frac)

    density_rgb = scalar_heatmap(density_display)
    panel_images: list[tuple[str, str, Image.Image | np.ndarray]] = [
        ("input", "Input (processed)", input_rgb),
        ("tensor_energy", "Tensor energy (robust log display)", scalar_heatmap(energy_display)),
        ("tensor_coherence", "Tensor coherence", scalar_heatmap(tensor.coherence)),
        ("tensor_labels", "Tensor classes: flat / edge / corner", labels_to_rgb(tensor.label)),
        (
            "tensor_tangents",
            "Tangents cyan; normals orange",
            _draw_tensor_glyphs(image, glyph_rows, cfg.glyph_length),
        ),
        ("sampling_density", "Sampling density (robust log display)", density_rgb),
        (
            "sampling_sites",
            f"{cfg.strategy} sites + unit-area metric shape",
            _sampling_panel(density_rgb, field, tensor, cfg),
        ),
        (
            "gaussian_ellipses",
            "Initialized RS Gaussians (1 sigma)",
            _gaussian_panel(image, field, cfg.ellipse_limit),
        ),
        (
            "initial_reconstruction",
            "Initial normalized reconstruction",
            np.round(np.clip(reconstruction, 0, 1) * 255).astype(np.uint8),
        ),
        (
            "initial_abs_error",
            "Initial mean absolute RGB error",
            scalar_heatmap(np.mean(np.abs(reconstruction - image), axis=2)),
        ),
        (
            "coverage_denominator",
            "Normalized-renderer denominator",
            scalar_heatmap(denominator_display),
        ),
        (
            "effective_contributors",
            "Effective contributor count",
            scalar_heatmap(effective_display),
        ),
        (
            "responsibility_entropy",
            "Responsibility entropy (nats)",
            scalar_heatmap(entropy_display),
        ),
        ("dominant_owner", "Dominant Gaussian ownership", _owner_rgb(diagnostics.dominant_owner)),
    ]

    saved_panels: list[tuple[str, Path]] = []
    for stem, title, panel in panel_images:
        path = output / f"{stem}.png"
        _save_png(path, panel)
        saved_panels.append((title, path))

    montage_path = output / "method_diagnostics_montage.png"
    disclaimer = (
        "Initialization diagnostics only — not optimized, held-out, or comparative evidence."
    )
    _save_png(montage_path, _montage(saved_panels, cfg.montage_columns, disclaimer))

    overview_path = output / "method_overview.svg"
    overview_path.write_text(_method_overview_svg(), encoding="utf-8")

    arrays = {
        "tensor_lam1": tensor.lam1,
        "tensor_lam2": tensor.lam2,
        "tensor_energy": tensor.energy,
        "tensor_coherence": tensor.coherence,
        "tensor_across_edge_angle": tensor.across_edge_angle,
        "tensor_along_edge_angle": tensor.along_edge_angle,
        "tensor_label": tensor.label,
        "sampling_density_pmf": density,
        "field_means_xy": field.means.detach().cpu().numpy(),
        "field_scales_sx_sy": field.effective_scales().detach().cpu().numpy(),
        "field_rotations_theta": field.rotations.detach().cpu().numpy(),
        "field_colors_rgb": field.colors.detach().cpu().numpy(),
        "initial_reconstruction_rgb": reconstruction,
        "coverage_denominator": diagnostics.denominator,
        "responsibility_sum": diagnostics.responsibility_sum,
        "effective_contributor_count": diagnostics.effective_count,
        "responsibility_entropy_nats": diagnostics.entropy,
        "active_contributor_count": diagnostics.active_count,
        "dominant_owner_index": diagnostics.dominant_owner,
        "tensor_glyph_rows": glyph_rows,
    }
    arrays_path = output / "diagnostics.npz"
    _deterministic_npz(arrays_path, arrays)

    source_info["source_path"] = _relative_source_path(source, root)
    resolved = {
        "schema_version": SCHEMA_VERSION,
        "figure": asdict(cfg),
        "structure_tensor": asdict(tensor_cfg),
        "initializer": asdict(init_cfg),
        "renderer": {
            "mode": "normalized",
            "sigma_cutoff": cfg.sigma_cutoff,
            "support_fade": False,
            "device": "cpu",
            "field_state": "initialization_only_no_fit",
        },
    }
    config_path = output / "config.json"
    _json_dump(config_path, resolved)

    output_paths = [path for _, path in saved_panels] + [
        montage_path,
        overview_path,
        arrays_path,
        config_path,
    ]
    covered = diagnostics.active_count > 0
    effective_violation = 0.0
    entropy_violation = 0.0
    if np.any(covered):
        effective_violation = float(
            np.max(
                np.maximum(
                    diagnostics.effective_count[covered] - diagnostics.active_count[covered], 0
                )
            )
        )
        entropy_violation = float(
            np.max(
                np.maximum(
                    diagnostics.entropy[covered]
                    - np.log(np.maximum(diagnostics.active_count[covered], 1)),
                    0,
                )
            )
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "explanatory initialization and normalized-renderer diagnostics",
        "claim_limit": disclaimer,
        "source": source_info,
        "resolved_config": resolved,
        "conventions": {
            "image": "(H,W,3) float32 RGB in [0,1] after optional crop/resize",
            "position": "(x,y) pixel coordinates; integer coordinates are pixel centers",
            "rotation": "theta is the sx axis; tensor-aligned sx follows the edge tangent",
            "sampling_metric_panel": (
                "ellipses show the unit-area local tensor metric shape at a fixed display scale; "
                "they are not Gaussian support ellipses or transmitted geometry"
            ),
            "gaussian_panel": "actual initialized RS covariance at one standard deviation",
            "effective_count": "(sum_i w_i)^2 / sum_i w_i^2",
            "responsibility_entropy": "-sum_i r_i log(r_i), r_i=w_i/sum_j w_j",
            "dominant_owner": "smallest Gaussian index wins exact weight ties",
        },
        "label_legend": {
            LABEL_NAMES[key]: {"id": key, "rgb": list(color)} for key, color in LABEL_COLORS.items()
        },
        "display_scaling": {
            "tensor_energy": energy_scale,
            "sampling_density": density_scale,
            "coverage_denominator": denominator_scale,
            "effective_contributors": effective_scale,
            "responsibility_entropy": entropy_scale,
            "note": "display transforms affect PNGs only; diagnostics.npz stores raw arrays",
        },
        "diagnostic_checks": {
            "covered_pixel_fraction": float(np.mean(covered)),
            "max_responsibility_sum_error_on_covered": (
                0.0
                if not np.any(covered)
                else float(np.max(np.abs(diagnostics.responsibility_sum[covered] - 1.0)))
            ),
            "max_effective_count_bound_violation": effective_violation,
            "max_entropy_bound_violation_nats": entropy_violation,
            "field_count": int(field.n),
            "glyph_count": int(len(glyph_rows)),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pillow": _package_version("Pillow"),
            "structsplat": _package_version("structsplat"),
        },
        "repository": _git_provenance(root),
        "implementation_sources": _implementation_provenance(root),
        "outputs": {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(output_paths)
        },
    }
    manifest_path = output / "manifest.json"
    _json_dump(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic StructSplat method/renderer diagnostic panels. "
            "Outputs are initialization-only explanatory figures, not benchmark evidence."
        )
    )
    parser.add_argument("image", type=Path, help="input image")
    parser.add_argument("--outdir", type=Path, required=True, help="output figure bundle")
    parser.add_argument(
        "--max-side", type=int, default=256, help="resize longest side (0 disables)"
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="crop in oriented source-image coordinates before resize",
    )
    parser.add_argument("--strategy", choices=STRATEGIES, default="aniso_onedge")
    parser.add_argument("--num-gaussians", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-oversample", type=float, default=4.0)
    parser.add_argument(
        "--density-mode",
        choices=("structure", "gradient", "variance", "hybrid", "uniform"),
        default="structure",
    )
    parser.add_argument("--density-base", type=float, default=0.05)
    parser.add_argument("--density-power", type=float, default=1.0)
    parser.add_argument("--max-axis-ratio", type=float, default=6.0)
    parser.add_argument("--coherence-power", type=float, default=1.0)
    parser.add_argument("--init-scale-mult", type=float, default=1.0)
    parser.add_argument(
        "--gradient-operator", choices=("central", "sobel", "scharr"), default="central"
    )
    parser.add_argument("--color-space", choices=("luma", "rgb"), default="luma")
    parser.add_argument("--grad-sigma", type=float, default=1.0)
    parser.add_argument("--tensor-sigma", type=float, default=2.0)
    parser.add_argument("--flat-frac", type=float, default=0.02)
    parser.add_argument("--corner-frac", type=float, default=0.15)
    parser.add_argument("--sigma-cutoff", type=float, default=3.0)
    parser.add_argument("--glyph-spacing", type=int, default=18)
    parser.add_argument("--glyph-length", type=float, default=8.0)
    parser.add_argument("--glyph-min-coherence", type=float, default=0.10)
    parser.add_argument("--ellipse-limit", type=int, default=220)
    parser.add_argument("--metric-ellipse-limit", type=int, default=72)
    parser.add_argument("--montage-columns", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = FigureConfig(
        max_side=None if args.max_side == 0 else args.max_side,
        crop=None if args.crop is None else tuple(args.crop),
        strategy=args.strategy,
        num_gaussians=args.num_gaussians,
        seed=args.seed,
        candidate_oversample=args.candidate_oversample,
        density_mode=args.density_mode,
        density_base=args.density_base,
        density_power=args.density_power,
        max_axis_ratio=args.max_axis_ratio,
        coherence_power=args.coherence_power,
        init_scale_mult=args.init_scale_mult,
        gradient_operator=args.gradient_operator,
        color_space=args.color_space,
        grad_sigma=args.grad_sigma,
        tensor_sigma=args.tensor_sigma,
        flat_frac=args.flat_frac,
        corner_frac=args.corner_frac,
        sigma_cutoff=args.sigma_cutoff,
        glyph_spacing=args.glyph_spacing,
        glyph_length=args.glyph_length,
        glyph_min_coherence=args.glyph_min_coherence,
        ellipse_limit=args.ellipse_limit,
        metric_ellipse_limit=args.metric_ellipse_limit,
        montage_columns=args.montage_columns,
    )
    manifest = generate_figure_bundle(args.image, args.outdir, config)
    print(
        f"wrote {args.outdir / 'method_diagnostics_montage.png'} and manifest.json "
        f"for {manifest['diagnostic_checks']['field_count']} initialized Gaussians"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
