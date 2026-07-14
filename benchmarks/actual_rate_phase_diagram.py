"""BENCH-007: actual-rate structure phase diagram and publication artifacts.

This harness is deliberately separate from ``benchmarks.rate_distortion``.  It
freezes source hashes and a resolution-normalized candidate grid before scores
are inspected, fits every arm/count independently, measures complete SSPL1
containers after cold decode, journals every cell, and selects candidates only
under exact byte caps.  Analysis emits the F5--F9 paper figures and an audit HTML
index; Stage-0 results remain plumbing/calibration evidence only.

Examples (run from the repository root with ``PYTHONPATH=src``)::

    python -m benchmarks.actual_rate_phase_diagram freeze \
      --stage stage0a --data-root tests/test_images \
      --images tests/test_images/*.jpg --bytes-per-gaussian 11 \
      --manifest results/bench007-stage0a/manifest.json --iters 2
    python -m benchmarks.actual_rate_phase_diagram run \
      --manifest results/bench007-stage0a/manifest.json \
      --data-root . --outdir results/bench007-stage0a --device cuda
    python -m benchmarks.actual_rate_phase_diagram analyze \
      --manifest results/bench007-stage0a/manifest.json \
      --data-root . --outdir results/bench007-stage0a
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


SCHEMA = "structsplat.bench007.v1"
MANIFEST_SCHEMA = "structsplat.bench007.manifest.v1"
DEFAULT_TARGETS = (0.25, 0.5, 1.0, 2.0, 4.0)
STAGE_TARGETS = {
    "stage0a": (0.5, 1.0),
    "stage1": (0.5, 1.0),
    "stage2": DEFAULT_TARGETS,
}
STAGE_IDS = {
    "stage0a": ("0009", "0025", "0030", "0034"),
    "stage0b": ("0002", "0268", "0534", "0800"),
    "stage1": ("0001", "0115", "0229", "0343", "0457", "0571", "0685", "0799"),
    "stage2": tuple(f"{index:04d}" for index in range(801, 901)),
}
DEFAULT_MULTIPLIERS = (0.75, 1.0, 4.0 / 3.0)
DEFAULT_BIT_MIXES = ((16, 8, 8, 8), (12, 8, 6, 8), (12, 6, 6, 6), (10, 5, 5, 5))
BENCHMARK_RENDERERS = ("normalized", "cuda")
PRIMARY_ARM = "tensor_wse"
ARMS: dict[str, dict[str, Any]] = {
    "tensor_wse": {
        "label": "Tensor-metric WSE",
        "strategy": "aniso_onedge",
        "density_mode": "structure",
        "sampling_mode": "wse",
        "orientation_mode": "tensor",
        "role": "candidate",
    },
    "quadtree_wse": {
        "label": "Quadtree WSE",
        "strategy": "quadtree_wse",
        "density_mode": "structure",
        "sampling_mode": "wse",
        "orientation_mode": "tensor",
        "role": "engineering_control",
    },
    "local_slic_sobel_control": {
        "label": "Local SLIC/Sobel control",
        "strategy": "local_slic_sobel_control",
        "density_mode": "uniform",
        "sampling_mode": "wse",
        "orientation_mode": "zero",
        "role": "direct_control",
    },
    "local_gradient_control": {
        "label": "Local gradient control",
        "strategy": "iso_blue_noise",
        "density_mode": "gradient",
        "sampling_mode": "density_random",
        "orientation_mode": "zero",
        "role": "direct_control",
    },
    "uniform_wse": {
        "label": "Uniform Euclidean WSE",
        "strategy": "iso_blue_noise",
        "density_mode": "uniform",
        "sampling_mode": "wse",
        "orientation_mode": "zero",
        "role": "direct_control",
    },
    "random": {
        "label": "Seeded random",
        "strategy": "random",
        "density_mode": "uniform",
        "sampling_mode": "density_random",
        "orientation_mode": "zero",
        "role": "direct_control",
    },
}
DIRECT_CONTROL_ARMS = tuple(name for name, spec in ARMS.items() if spec["role"] == "direct_control")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            # A killed process can leave only its final append incomplete. Earlier or
            # newline-terminated corruption is not safe to ignore.
            if line_number == len(lines) and not line.endswith(("\n", "\r")):
                break
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def repository_provenance() -> dict[str, Any]:
    diff = _git_output("diff", "--binary", "HEAD")
    untracked = (_git_output("ls-files", "--others", "--exclude-standard") or "").splitlines()
    untracked_hashes = {
        name: sha256_file(name) for name in sorted(untracked) if Path(name).is_file()
    }
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "dirty": bool((_git_output("status", "--porcelain") or "").strip()),
        "tracked_diff_sha256": _sha256_bytes((diff or "").encode("utf-8")),
        "untracked_file_sha256": untracked_hashes,
    }


def environment_provenance() -> dict[str, Any]:
    try:
        import PIL
        pillow_version = PIL.__version__
    except Exception:  # pragma: no cover
        pillow_version = None
    try:
        import skimage
        skimage_version = skimage.__version__
    except Exception:
        skimage_version = None
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:
        scipy_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": pillow_version,
        "scikit_image": skimage_version,
        "scipy": scipy_version,
    }


def decoded_rgb_sha256(path: str | Path) -> tuple[str, int, int]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    prefix = np.asarray([height, width, 3], dtype="<u8").tobytes()
    return _sha256_bytes(prefix + rgb.tobytes()), int(height), int(width)


def load_native_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def candidate_count(
    pixel_count: int,
    target_bpp: float,
    bytes_per_gaussian: float,
    multiplier: float,
    *,
    min_gaussians: int = 64,
    max_gaussians_per_pixel: float = 0.25,
) -> int:
    """Resolution-normalized, half-up-rounded count estimate for one bracket."""
    if pixel_count <= 0 or target_bpp <= 0.0 or bytes_per_gaussian <= 0.0 or multiplier <= 0.0:
        raise ValueError("pixel_count, target_bpp, bytes_per_gaussian, and multiplier must be > 0")
    estimate = target_bpp * pixel_count / (8.0 * bytes_per_gaussian)
    rounded = int(math.floor(multiplier * estimate + 0.5))
    maximum = max(min_gaussians, int(math.ceil(max_gaussians_per_pixel * pixel_count)))
    return min(maximum, max(min_gaussians, rounded))


def candidate_ladder(
    pixel_count: int,
    targets: Sequence[float],
    bytes_per_gaussian: float,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    *,
    min_gaussians: int = 64,
    max_gaussians_per_pixel: float = 0.25,
) -> list[int]:
    return sorted({
        candidate_count(
            pixel_count,
            float(target),
            bytes_per_gaussian,
            float(multiplier),
            min_gaussians=min_gaussians,
            max_gaussians_per_pixel=max_gaussians_per_pixel,
        )
        for target in targets
        for multiplier in multipliers
    })


def exact_byte_cap(target_bpp: float, height: int, width: int) -> int:
    """Largest complete-byte stream satisfying the declared bpp cap."""
    if target_bpp <= 0.0 or height <= 0 or width <= 0:
        raise ValueError("target_bpp and image dimensions must be positive")
    return int(math.floor(float(target_bpp) * height * width / 8.0 + 1e-12))


def actual_bpp(num_bytes: int, height: int, width: int) -> float:
    if num_bytes < 0 or height <= 0 or width <= 0:
        raise ValueError("num_bytes must be nonnegative and dimensions positive")
    return 8.0 * int(num_bytes) / float(height * width)


def _relative_to_root(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"image {resolved} is outside data root {root.resolve()}") from exc


def _infer_image_id(path: Path) -> str:
    stem = path.stem
    digits = "".join(character for character in stem if character.isdigit())
    return digits[-4:] if len(digits) >= 4 else stem


def freeze_manifest(
    image_paths: Sequence[str | Path],
    data_root: str | Path,
    output_path: str | Path,
    *,
    stage: str,
    bytes_per_gaussian: float,
    targets: Sequence[float] | None = None,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    bit_mixes: Sequence[Sequence[int]] = DEFAULT_BIT_MIXES,
    arms: Sequence[str] = tuple(ARMS),
    seed: int = 0,
    iters: int = 1500,
    qat_iters: int = 0,
    renderer: str = "normalized",
    render_chunk: int = 512,
    min_gaussians: int = 64,
    max_gaussians_per_pixel: float = 0.25,
    compute_lpips: bool = True,
    enforce_stage_ids: bool = True,
) -> dict[str, Any]:
    if stage not in ("stage0a", "stage1", "stage2"):
        raise ValueError("freeze stage must be stage0a, stage1, or stage2")
    if renderer not in BENCHMARK_RENDERERS:
        raise ValueError(
            f"BENCH-007 renderer must be one of {BENCHMARK_RENDERERS}, got {renderer!r}"
        )
    targets = tuple(float(value) for value in (targets or STAGE_TARGETS[stage]))
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise ValueError(f"unknown benchmark arms: {unknown}")
    if stage == "stage1" and set(arms) != set(ARMS):
        raise ValueError("Stage 1 requires all six preregistered common-renderer arms")
    root = Path(data_root)
    images: list[dict[str, Any]] = []
    for raw_path in sorted({str(Path(value)) for value in image_paths}):
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        rgb_hash, height, width = decoded_rgb_sha256(path)
        images.append({
            "id": _infer_image_id(path),
            "path": _relative_to_root(path, root),
            "source_sha256": sha256_file(path),
            "decoded_rgb_sha256": rgb_hash,
            "height": height,
            "width": width,
            "pixel_count": height * width,
            "candidate_counts": candidate_ladder(
                height * width,
                targets,
                bytes_per_gaussian,
                multipliers,
                min_gaussians=min_gaussians,
                max_gaussians_per_pixel=max_gaussians_per_pixel,
            ),
        })
    if not images:
        raise ValueError("cannot freeze an empty image manifest")
    ids = [entry["id"] for entry in images]
    if len(ids) != len(set(ids)):
        raise ValueError(f"image IDs are not unique: {ids}")
    expected_ids = set(STAGE_IDS[stage])
    if enforce_stage_ids and set(ids) != expected_ids:
        raise ValueError(
            f"{stage} requires frozen IDs {sorted(expected_ids)}, got {sorted(ids)}"
        )
    if not enforce_stage_ids and stage != "stage0a":
        raise ValueError("stage-ID enforcement may be disabled only for Stage-0a plumbing tests")

    from structsplat.structural_controls import SlicSobelConfig

    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "stage": stage,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": (
            "plumbing_only" if stage == "stage0a" else
            "development_killing_pilot" if stage == "stage1" else
            "held_out_confirmation"
        ),
        "stage_ids_enforced": bool(enforce_stage_ids),
        "images": images,
        "targets_bpp": list(targets),
        "seed": int(seed),
        "arms": [{"name": arm, **ARMS[arm]} for arm in arms],
        "candidate_formula": {
            "estimate": "target_bpp*original_pixels/(8*bytes_per_gaussian)",
            "rounding": "floor(multiplier*estimate+0.5)",
            "bytes_per_gaussian": float(bytes_per_gaussian),
            "multipliers": [float(value) for value in multipliers],
            "min_gaussians": int(min_gaussians),
            "max_gaussians_per_pixel": float(max_gaussians_per_pixel),
            "absolute_count_ceiling": None,
        },
        "bit_mixes": [list(map(int, mix)) for mix in bit_mixes],
        "init_common": {
            "candidate_oversample": 6.0,
            "scale_mode": "spacing",
            "color_mode": "bilinear",
            "opacity_mode": "none",
            "seed": int(seed),
        },
        "fit_config": {
            "iters": int(iters),
            "renderer": renderer,
            "render_chunk": int(render_chunk),
            "checkpoint_policy": "best_psnr_final_count",
            "compute_lpips": False,
        },
        "renderer_protocol": {
            "equation": "normalized_weighted_sum",
            "implementation": (
                "pytorch_reference" if renderer == "normalized" else "owned_exact_cuda"
            ),
            "cuda_is_equation_preserving": renderer == "cuda",
            "required_cold_parity_atol": 1e-6,
        },
        "codec_config": {
            "stream_codec": "zlib",
            "zlib_level": 9,
            "morton_reorder": True,
            "color_delta": True,
            "means_byte_planar": True,
            "rot_circular": True,
        },
        "qat_iters": int(qat_iters),
        "central_scoring": {
            "clamp_decoded_rgb": True,
            "psnr": True,
            "ssim": True,
            "ms_ssim": True,
            "lpips_on_selected": bool(compute_lpips),
            "cold_parity_atol": 1e-6,
        },
        "mechanism_protocol": {
            "luma": "rec709",
            "edge_band": "top_positive_decile_target_sobel_dilated_radius_2",
            "texture_band": "top_quartile_5x5_target_luma_variance_excluding_edge_band",
            "bleed": "mean_abs_target_contrast_minus_aligned_prediction_at_1.5px_normal_offsets",
            "texture_failure_relative_threshold": 0.05,
        },
        "slic_sobel_control": SlicSobelConfig().provenance(),
        "repository": repository_provenance(),
        "environment": environment_provenance(),
    }
    payload["manifest_sha256"] = _sha256_bytes(_canonical_json(payload))
    _atomic_json(Path(output_path), payload)
    return payload


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported BENCH-007 manifest schema: {payload.get('schema')!r}")
    claimed = payload.get("manifest_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual = _sha256_bytes(_canonical_json(unhashed))
    if claimed != actual:
        raise ValueError(f"manifest hash mismatch: claimed {claimed}, actual {actual}")
    return payload


def validate_manifest_sources(manifest: dict[str, Any], data_root: str | Path) -> None:
    root = Path(data_root)
    for entry in manifest["images"]:
        path = root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != entry["source_sha256"]:
            raise ValueError(f"source file hash changed for {entry['id']}: {path}")
        rgb_hash, height, width = decoded_rgb_sha256(path)
        if rgb_hash != entry["decoded_rgb_sha256"]:
            raise ValueError(f"decoded RGB hash changed for {entry['id']}: {path}")
        if [height, width] != [entry["height"], entry["width"]]:
            raise ValueError(
                f"source dimensions changed for {entry['id']}: "
                f"expected {(entry['height'], entry['width'])}, got {(height, width)}"
            )


def monotone_envelope(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the measured nondominated rate/PSNR envelope without interpolation."""
    valid = [
        row for row in rows
        if row.get("status", "ok") == "ok"
        and row.get("bytes") is not None
        and row.get("psnr") is not None
        and np.isfinite(float(row["psnr"]))
    ]
    valid.sort(key=lambda row: (int(row["bytes"]), -float(row["psnr"])))
    by_bytes: list[dict[str, Any]] = []
    for row in valid:
        if by_bytes and int(row["bytes"]) == int(by_bytes[-1]["bytes"]):
            continue
        by_bytes.append(row)
    envelope: list[dict[str, Any]] = []
    best = -math.inf
    for row in by_bytes:
        quality = float(row["psnr"])
        if quality > best + 1e-12:
            envelope.append(row)
            best = quality
    return envelope


def select_under_cap(
    rows: Iterable[dict[str, Any]], target_bpp: float, height: int, width: int
) -> dict[str, Any] | None:
    cap = exact_byte_cap(target_bpp, height, width)
    feasible = [
        row for row in rows
        if row.get("status", "ok") == "ok"
        and row.get("bytes") is not None
        and int(row["bytes"]) <= cap
        and row.get("psnr") is not None
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda row: (float(row["psnr"]), -int(row["bytes"]), -int(row["n_gaussians"])),
    )


def bd_rate(
    candidate: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    *,
    min_points: int = 4,
) -> dict[str, Any]:
    """PCHIP Bjontegaard delta-rate over the common measured PSNR interval only."""
    from scipy.interpolate import PchipInterpolator

    def curve(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        envelope = monotone_envelope(rows)
        # Invert PSNR->rate; if numerical ties occur, retain the lowest measured rate.
        pairs: dict[float, float] = {}
        for row in envelope:
            quality = float(row["psnr"])
            rate = float(row.get("bpp", actual_bpp(row["bytes"], row["height"], row["width"])))
            pairs[quality] = min(rate, pairs.get(quality, math.inf))
        qualities = np.asarray(sorted(pairs), dtype=np.float64)
        rates = np.asarray([pairs[value] for value in qualities], dtype=np.float64)
        return qualities, rates

    q_c, r_c = curve(candidate)
    q_r, r_r = curve(reference)
    if len(q_c) < min_points or len(q_r) < min_points:
        return {"defined": False, "reason": "fewer_than_four_envelope_points"}
    lower = max(float(q_c.min()), float(q_r.min()))
    upper = min(float(q_c.max()), float(q_r.max()))
    if not upper > lower:
        return {"defined": False, "reason": "no_common_measured_psnr_interval"}
    overlap_c = int(np.count_nonzero((q_c >= lower) & (q_c <= upper)))
    overlap_r = int(np.count_nonzero((q_r >= lower) & (q_r <= upper)))
    if min(overlap_c, overlap_r) < min_points:
        return {
            "defined": False,
            "reason": "fewer_than_four_points_in_common_interval",
            "overlap_points": [overlap_c, overlap_r],
            "psnr_interval": [lower, upper],
        }
    candidate_curve = PchipInterpolator(q_c, np.log(r_c), extrapolate=False)
    reference_curve = PchipInterpolator(q_r, np.log(r_r), extrapolate=False)
    integral_c = float(candidate_curve.integrate(lower, upper))
    integral_r = float(reference_curve.integrate(lower, upper))
    delta = (math.exp((integral_c - integral_r) / (upper - lower)) - 1.0) * 100.0
    return {
        "defined": True,
        "bd_rate_percent": delta,
        "psnr_interval": [lower, upper],
        "overlap_points": [overlap_c, overlap_r],
        "no_extrapolation": True,
    }


def _sobel_xy(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float64)
    padded = np.pad(image, 1, mode="reflect")
    kernel_x = np.array(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64
    ) / 8.0
    gradient_x = np.zeros_like(image)
    gradient_y = np.zeros_like(image)
    for iy in range(3):
        for ix in range(3):
            window = padded[iy:iy + image.shape[0], ix:ix + image.shape[1]]
            gradient_x += kernel_x[iy, ix] * window
            gradient_y += kernel_x[ix, iy] * window
    return gradient_x, gradient_y


def _dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("dilation radius must be >= 0")
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, radius, mode="constant")
    out = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return out


def _box_mean(image: np.ndarray, radius: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if radius < 0:
        raise ValueError("box radius must be >= 0")
    if radius == 0:
        return image.copy()
    padded = np.pad(image, radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    size = 2 * radius + 1
    sums = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return sums / float(size * size)


def mechanism_masks(target: np.ndarray) -> dict[str, np.ndarray | float]:
    """Predeclared edge and texture bands derived only from the target image."""
    if target.ndim != 3 or target.shape[2] != 3:
        raise ValueError(f"expected HWC RGB target, got {target.shape}")
    luma = np.asarray(target, dtype=np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    gradient_x, gradient_y = _sobel_xy(luma)
    magnitude = np.hypot(gradient_x, gradient_y)
    positive = magnitude[magnitude > 0.0]
    edge_threshold = float(np.quantile(positive, 0.9)) if positive.size else math.inf
    edge_core = magnitude >= edge_threshold
    edge_band = _dilate_square(edge_core, 2)

    mean = _box_mean(luma, 2)
    variance = np.maximum(_box_mean(np.square(luma), 2) - np.square(mean), 0.0)
    eligible = ~edge_band
    eligible_values = variance[eligible]
    texture_threshold = (
        float(np.quantile(eligible_values, 0.75)) if eligible_values.size else math.inf
    )
    texture_band = eligible & (variance >= texture_threshold)
    return {
        "luma": luma,
        "gradient_x": gradient_x,
        "gradient_y": gradient_y,
        "gradient_magnitude": magnitude,
        "edge_core": edge_core,
        "edge_band": edge_band,
        "texture_variance": variance,
        "texture_band": texture_band,
        "edge_threshold": edge_threshold,
        "texture_threshold": texture_threshold,
    }


def _bilinear_scalar(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x = np.clip(x, 0.0, width - 1.0)
    y = np.clip(y, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = x - x0
    fy = y - y0
    return (
        image[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + image[y0, x1] * fx * (1.0 - fy)
        + image[y1, x0] * (1.0 - fx) * fy
        + image[y1, x1] * fx * fy
    )


def mechanism_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    masks: dict[str, np.ndarray | float] | None = None,
) -> dict[str, float | int | None]:
    """Edge/texture RGB MSE and signed target-normal cross-edge contrast loss."""
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    masks = masks or mechanism_masks(target)
    edge_band = np.asarray(masks["edge_band"], dtype=bool)
    texture_band = np.asarray(masks["texture_band"], dtype=bool)
    squared = np.square(np.asarray(prediction, dtype=np.float64) - target).mean(axis=2)

    def band_mean(mask: np.ndarray) -> float | None:
        return float(squared[mask].mean()) if mask.any() else None

    target_luma = np.asarray(masks["luma"], dtype=np.float64)
    prediction_luma = np.asarray(prediction, dtype=np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    gradient_x = np.asarray(masks["gradient_x"], dtype=np.float64)
    gradient_y = np.asarray(masks["gradient_y"], dtype=np.float64)
    magnitude = np.asarray(masks["gradient_magnitude"], dtype=np.float64)
    core = np.asarray(masks["edge_core"], dtype=bool) & (magnitude > 1e-12)
    bleed: float | None = None
    target_contrast: float | None = None
    prediction_aligned_contrast: float | None = None
    if core.any():
        y, x = np.nonzero(core)
        nx = gradient_x[core] / magnitude[core]
        ny = gradient_y[core] / magnitude[core]
        offset = 1.5
        target_delta = _bilinear_scalar(target_luma, x + offset * nx, y + offset * ny) - (
            _bilinear_scalar(target_luma, x - offset * nx, y - offset * ny)
        )
        prediction_delta = _bilinear_scalar(
            prediction_luma, x + offset * nx, y + offset * ny
        ) - _bilinear_scalar(prediction_luma, x - offset * nx, y - offset * ny)
        sign = np.where(target_delta >= 0.0, 1.0, -1.0)
        aligned = sign * prediction_delta
        target_abs = np.abs(target_delta)
        bleed = float(np.mean(target_abs - aligned))
        target_contrast = float(np.mean(target_abs))
        prediction_aligned_contrast = float(np.mean(aligned))
    return {
        "edge_mse": band_mean(edge_band),
        "texture_mse": band_mean(texture_band),
        "signed_cross_edge_bleed": bleed,
        "target_cross_edge_contrast": target_contrast,
        "prediction_aligned_cross_edge_contrast": prediction_aligned_contrast,
        "edge_pixels": int(edge_band.sum()),
        "texture_pixels": int(texture_band.sum()),
        "edge_fraction": float(edge_band.mean()),
        "texture_fraction": float(texture_band.mean()),
    }


def cluster_bootstrap(
    values: Sequence[float], *, samples: int = 10_000, seed: int = 7007
) -> dict[str, Any]:
    clean = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=np.float64)
    if clean.size == 0:
        return {"n_images": 0, "mean": None, "ci95": [None, None], "p_two_sided": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, clean.size, size=(samples, clean.size))
    distribution = clean[indices].mean(axis=1)
    tail = min(float(np.mean(distribution <= 0.0)), float(np.mean(distribution >= 0.0)))
    return {
        "n_images": int(clean.size),
        "mean": float(clean.mean()),
        "ci95": [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))],
        "p_two_sided": min(1.0, 2.0 * tail),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def holm_adjust(p_values: dict[str, float | None]) -> dict[str, float | None]:
    finite = sorted(
        ((key, float(value)) for key, value in p_values.items() if value is not None),
        key=lambda pair: pair[1],
    )
    adjusted: dict[str, float | None] = {key: None for key in p_values}
    running = 0.0
    count = len(finite)
    for rank, (key, value) in enumerate(finite):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[key] = running
    return adjusted


def _arm_names(manifest: dict[str, Any]) -> list[str]:
    return [entry["name"] for entry in manifest["arms"]]


def fit_cell_key(manifest_hash: str, image_id: str, arm: str, count: int) -> str:
    return _sha256_bytes(_canonical_json([manifest_hash, image_id, arm, int(count)]))[:24]


def candidate_cell_key(fit_key: str, bit_mix: Sequence[int], qat_iters: int) -> str:
    return _sha256_bytes(_canonical_json([fit_key, list(map(int, bit_mix)), int(qat_iters)]))[:24]


def dry_run_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    fits = sum(len(image["candidate_counts"]) for image in manifest["images"]) * len(
        manifest["arms"]
    )
    candidates = fits * len(manifest["bit_mixes"])
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "stage": manifest["stage"],
        "images": len(manifest["images"]),
        "arms": len(manifest["arms"]),
        "fits": fits,
        "codec_candidates": candidates,
        "targets": len(manifest["targets_bpp"]),
        "rdo_selections": len(manifest["images"]) * len(manifest["arms"])
        * len(manifest["targets_bpp"]),
        "per_image": {
            image["id"]: {
                "pixels": image["pixel_count"],
                "candidate_counts": image["candidate_counts"],
                "fits": len(image["candidate_counts"]) * len(manifest["arms"]),
            }
            for image in manifest["images"]
        },
    }


def _latest_rows(path: Path, key_name: str) -> dict[str, dict[str, Any]]:
    return {str(row[key_name]): row for row in read_jsonl(path) if key_name in row}


def _process_peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":  # bytes on macOS, KiB on Linux
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _gpu_sync(torch_module: Any, device: str) -> None:
    if str(device).startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(device)


def _gpu_peak_mb(torch_module: Any, device: str) -> dict[str, float | None]:
    if not str(device).startswith("cuda") or not torch_module.cuda.is_available():
        return {"peak_vram_allocated_mb": None, "peak_vram_reserved_mb": None}
    return {
        "peak_vram_allocated_mb": torch_module.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        "peak_vram_reserved_mb": torch_module.cuda.max_memory_reserved(device) / (1024.0 ** 2),
    }


def _reset_gpu_peak(torch_module: Any, device: str) -> None:
    if str(device).startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats(device)


def _save_field_atomic(field: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp-{os.getpid()}.npz")
    field.save(str(temporary))
    os.replace(temporary, path)


def _render_decoded_field(field: Any, header: dict[str, Any]) -> Any:
    from structsplat.render import render_field

    height, width = int(header["H"]), int(header["W"])
    aa = float(header.get("aa_dilation", 0.0))
    sigma = float(header.get("sigma_cutoff", 3.0))
    return render_field(
        field.means,
        field.conics(aa),
        field.colors,
        field.radii(sigma, aa),
        height,
        width,
        int(header.get("render_chunk", 512)),
        header.get("renderer", "normalized"),
        field.opacity_values(),
        scales=field.scales(),
        rotations=field.rotations,
        support_fade=bool(header.get("support_fade", False)),
        sigma_cutoff=sigma,
    )


def _decoded_field_state_sha256(field: Any) -> str:
    """Hash the transmitted, decoded field state independently of renderer execution order."""
    digest = hashlib.sha256()
    for name in ("means", "log_scales", "rotations", "colors", "opacities"):
        value = getattr(field, name)
        digest.update(name.encode("utf-8") + b"\0")
        if value is None:
            digest.update(b"none\0")
            continue
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(_canonical_json(list(array.shape)) + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _decoded_field_state_max_abs(reference: Any, cold: Any) -> float:
    """Compare two decoded SSPL1 fields without invoking a nondeterministic CUDA render."""
    maximum = 0.0
    for name in ("means", "log_scales", "rotations", "colors", "opacities"):
        expected = getattr(reference, name)
        actual = getattr(cold, name)
        if (expected is None) != (actual is None):
            raise RuntimeError(f"decoded field optional-state mismatch for {name}")
        if expected is None:
            continue
        if expected.shape != actual.shape:
            raise RuntimeError(
                f"decoded field shape mismatch for {name}: {expected.shape} != {actual.shape}"
            )
        if expected.numel():
            maximum = max(maximum, float((expected - actual).abs().max().cpu()))
    return maximum


def _central_metrics(prediction: Any, target: Any) -> dict[str, float]:
    from structsplat import metrics

    prediction = prediction.clamp(0.0, 1.0)
    return {
        "psnr": float(metrics.psnr(prediction, target)),
        "ssim": float(metrics.ssim(prediction, target)),
        "ms_ssim": float(metrics.ms_ssim(prediction, target)),
    }


def _resolved_init_kwargs(manifest: dict[str, Any], arm: str, count: int) -> dict[str, Any]:
    common = dict(manifest["init_common"])
    spec = ARMS[arm]
    return {
        **common,
        "strategy": spec["strategy"],
        "density_mode": spec["density_mode"],
        "sampling_mode": spec["sampling_mode"],
        "orientation_mode": spec["orientation_mode"],
        "num_gaussians": int(count),
    }


def _write_run_snapshot(
    manifest: dict[str, Any], manifest_path: Path, outdir: Path, device: str
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    frozen_copy = outdir / "frozen_manifest.json"
    if frozen_copy.exists():
        existing = load_manifest(frozen_copy)
        if existing["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError(f"output directory belongs to a different manifest: {outdir}")
    else:
        shutil.copyfile(manifest_path, frozen_copy)
    snapshot = {
        "schema": SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "started_or_resumed_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "runtime_repository": repository_provenance(),
        "runtime_environment": environment_provenance(),
        "argv": sys.argv,
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda"] = torch.version.cuda
        if device.startswith("cuda") and torch.cuda.is_available():
            index = torch.device(device).index or 0
            snapshot["gpu"] = torch.cuda.get_device_name(index)
    except Exception:
        pass
    _atomic_json(outdir / "run_snapshot.json", snapshot)


def run_manifest(
    manifest_path: str | Path,
    data_root: str | Path,
    outdir: str | Path,
    *,
    device: str | None = None,
    only_arms: Sequence[str] | None = None,
    only_images: Sequence[str] | None = None,
    max_new_fits: int | None = None,
    retry_failed: bool = False,
    revalidate_candidates: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute or resume a frozen BENCH-007 manifest, journaling each fit/candidate."""
    import torch
    from structsplat import codec
    from structsplat import init as init_ops
    from structsplat.config import FitConfig, InitConfig
    from structsplat.fit import fit
    from structsplat.gaussians import GaussianField

    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    validate_manifest_sources(manifest, data_root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if manifest["fit_config"]["renderer"] == "cuda" and not str(device).startswith("cuda"):
        raise ValueError("a manifest frozen with renderer='cuda' requires a CUDA device")
    output = Path(outdir)
    _write_run_snapshot(manifest, manifest_path, output, device)
    fit_journal = output / "journals" / "fits.jsonl"
    candidate_journal = output / "journals" / "candidates.jsonl"
    error_journal = output / "journals" / "errors.jsonl"
    existing_fits = _latest_rows(fit_journal, "fit_key")
    existing_candidates = _latest_rows(candidate_journal, "candidate_key")
    allowed_arms = set(only_arms or _arm_names(manifest))
    unknown_arms = allowed_arms - set(_arm_names(manifest))
    if unknown_arms:
        raise ValueError(f"requested arms are absent from manifest: {sorted(unknown_arms)}")
    allowed_images = set(only_images or [entry["id"] for entry in manifest["images"]])

    completed_fits = 0
    completed_candidates = 0
    new_fits = 0
    for image_entry in manifest["images"]:
        image_id = image_entry["id"]
        if image_id not in allowed_images:
            continue
        image_path = Path(data_root) / image_entry["path"]
        image = load_native_rgb(image_path)
        height, width = image.shape[:2]
        if (height, width) != (image_entry["height"], image_entry["width"]):
            raise ValueError(f"native denominator mismatch for {image_id}")
        target = torch.as_tensor(image, dtype=torch.float32, device=device)
        masks = mechanism_masks(image)
        for arm in _arm_names(manifest):
            if arm not in allowed_arms:
                continue
            for count in image_entry["candidate_counts"]:
                key = fit_cell_key(manifest["manifest_sha256"], image_id, arm, count)
                field_path = output / "fields" / image_id / arm / f"n{count}-{key}.npz"
                initial_path = output / "initial_fields" / image_id / arm / f"n{count}-{key}.npz"
                history_path = output / "histories" / image_id / arm / f"n{count}-{key}.json"
                previous = existing_fits.get(key)
                field = None
                fit_row: dict[str, Any] | None = None
                reusable = bool(
                    previous
                    and previous.get("status") == "ok"
                    and field_path.is_file()
                    and previous.get("field_sha256") == sha256_file(field_path)
                )
                if reusable:
                    field = GaussianField.load(str(field_path), device=device)
                    fit_row = previous
                    completed_fits += 1
                elif previous and previous.get("status") == "failed" and not retry_failed:
                    continue
                else:
                    if max_new_fits is not None and new_fits >= max_new_fits:
                        return {
                            "status": "partial_limit_reached",
                            "new_fits": new_fits,
                            "completed_fits": completed_fits,
                            "completed_candidates": completed_candidates,
                        }
                    init_kwargs = _resolved_init_kwargs(manifest, arm, count)
                    fcfg = FitConfig(**manifest["fit_config"])
                    _reset_gpu_peak(torch, device)
                    start = time.perf_counter()
                    try:
                        initial = init_ops.build_field(
                            image, InitConfig(**init_kwargs), device=device
                        )
                        _gpu_sync(torch, device)
                        init_seconds = time.perf_counter() - start
                        _save_field_atomic(initial, initial_path)
                        fit_start = time.perf_counter()
                        result = fit(initial, target, fcfg, verbose=False)
                        _gpu_sync(torch, device)
                        fit_wall_seconds = time.perf_counter() - fit_start
                        field = result["field"]
                        _save_field_atomic(field, field_path)
                        _atomic_json(history_path, result["history"])
                        fit_row = {
                            "schema": SCHEMA,
                            "status": "ok",
                            "manifest_sha256": manifest["manifest_sha256"],
                            "fit_key": key,
                            "image_id": image_id,
                            "arm": arm,
                            "n_gaussians": int(count),
                            "height": height,
                            "width": width,
                            "source_sha256": image_entry["source_sha256"],
                            "decoded_rgb_sha256": image_entry["decoded_rgb_sha256"],
                            "init_config": init_kwargs,
                            "fit_config": manifest["fit_config"],
                            "init_seconds": init_seconds,
                            "fit_seconds": float(result["fit_seconds"]),
                            "fit_wall_seconds": fit_wall_seconds,
                            "fit_plus_init_seconds": init_seconds + fit_wall_seconds,
                            "fit_psnr": float(result["psnr"]),
                            "fit_ssim": float(result["ssim"]),
                            "fit_ms_ssim": float(result["ms_ssim"]),
                            "iterations_run": int(result["iterations_run"]),
                            "selected_iter": int(result["selected_iter"]),
                            "field_path": str(field_path.relative_to(output)),
                            "field_sha256": sha256_file(field_path),
                            "initial_field_path": str(initial_path.relative_to(output)),
                            "initial_field_sha256": sha256_file(initial_path),
                            "history_path": str(history_path.relative_to(output)),
                            "history_sha256": sha256_file(history_path),
                            "process_peak_rss_mb": _process_peak_rss_mb(),
                            **_gpu_peak_mb(torch, device),
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        }
                        _append_jsonl(fit_journal, fit_row)
                        existing_fits[key] = fit_row
                        new_fits += 1
                        completed_fits += 1
                        if verbose:
                            print(
                                f"fit {image_id} {arm} N={count}: "
                                f"{fit_row['fit_psnr']:.3f} dB, {fit_wall_seconds:.2f}s",
                                flush=True,
                            )
                    except Exception as exc:
                        failed = {
                            "schema": SCHEMA,
                            "status": "failed",
                            "manifest_sha256": manifest["manifest_sha256"],
                            "fit_key": key,
                            "image_id": image_id,
                            "arm": arm,
                            "n_gaussians": int(count),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                        }
                        _append_jsonl(fit_journal, failed)
                        _append_jsonl(error_journal, failed)
                        existing_fits[key] = failed
                        if verbose:
                            print(f"FAILED fit {image_id} {arm} N={count}: {exc}", flush=True)
                        continue

                if field is None or fit_row is None:
                    continue
                fcfg = FitConfig(**manifest["fit_config"])
                for mix in manifest["bit_mixes"]:
                    candidate_key = candidate_cell_key(key, mix, manifest["qat_iters"])
                    stream_path = (
                        output / "streams" / "candidates" / image_id / arm
                        / f"n{count}-b{'-'.join(map(str, mix))}-{candidate_key}.sspl1"
                    )
                    prior_candidate = existing_candidates.get(candidate_key)
                    reusable_candidate = bool(
                        prior_candidate
                        and prior_candidate.get("status") == "ok"
                        and stream_path.is_file()
                        and prior_candidate.get("stream_sha256") == sha256_file(stream_path)
                    )
                    if reusable_candidate and not revalidate_candidates:
                        completed_candidates += 1
                        continue
                    if (
                        prior_candidate
                        and prior_candidate.get("status") == "failed"
                        and not retry_failed
                        and not revalidate_candidates
                    ):
                        continue
                    try:
                        _reset_gpu_peak(torch, device)
                        previous_stream_sha256 = (
                            sha256_file(stream_path) if stream_path.is_file() else None
                        )
                        candidate_field = field.detached()
                        codec_config = codec.CodecConfig(
                            bits_means=int(mix[0]),
                            bits_scales=int(mix[1]),
                            bits_rot=int(mix[2]),
                            bits_colors=int(mix[3]),
                            **manifest["codec_config"],
                        )
                        qat_seconds = 0.0
                        if int(manifest["qat_iters"]) > 0:
                            qat_start = time.perf_counter()
                            codec_config = codec.qat_finetune(
                                candidate_field,
                                target,
                                fcfg,
                                codec_config,
                                iters=int(manifest["qat_iters"]),
                            )
                            _gpu_sync(torch, device)
                            qat_seconds = time.perf_counter() - qat_start
                        encode_start = time.perf_counter()
                        blob = codec.encode(candidate_field, height, width, codec_config, fcfg)
                        encode_seconds = time.perf_counter() - encode_start
                        _atomic_bytes(stream_path, blob)
                        header = codec.blob_header(blob)
                        if (int(header["H"]), int(header["W"]), int(header["n"])) != (
                            height, width, candidate_field.n
                        ):
                            raise RuntimeError("cold stream header does not match frozen target")

                        in_memory_decode_start = time.perf_counter()
                        in_memory_decoded_field = codec.decode(blob, device=device)
                        _gpu_sync(torch, device)
                        in_memory_decode_seconds = time.perf_counter() - in_memory_decode_start

                        # Validate the persisted stream against the in-memory encoded bytes at the
                        # decoded-field boundary. Rendering the same field twice with the exact
                        # CUDA implementation is not a valid parity oracle: atomic accumulation
                        # order can differ by a few float32 ulps between identical launches.
                        cold_start = time.perf_counter()
                        cold_blob = stream_path.read_bytes()
                        if cold_blob != blob:
                            raise RuntimeError("persisted stream bytes differ from encoded bytes")
                        decode_start = time.perf_counter()
                        cold_decoded_field = codec.decode(cold_blob, device=device)
                        _gpu_sync(torch, device)
                        decode_seconds = time.perf_counter() - decode_start
                        reference_field_sha256 = _decoded_field_state_sha256(
                            in_memory_decoded_field
                        )
                        cold_field_sha256 = _decoded_field_state_sha256(cold_decoded_field)
                        parity_max_abs = _decoded_field_state_max_abs(
                            in_memory_decoded_field, cold_decoded_field
                        )
                        parity_atol = float(
                            manifest["central_scoring"].get("cold_parity_atol", 1e-6)
                        )
                        if parity_max_abs > parity_atol or (
                            reference_field_sha256 != cold_field_sha256
                        ):
                            raise RuntimeError(
                                "cold/in-memory decoded field mismatch: "
                                f"max_abs={parity_max_abs} > {parity_atol} or state hash differs"
                            )
                        render_start = time.perf_counter()
                        cold_render = _render_decoded_field(cold_decoded_field, header)
                        _gpu_sync(torch, device)
                        render_seconds = time.perf_counter() - render_start
                        cold_decode_render_seconds = time.perf_counter() - cold_start
                        display_render = cold_render.clamp(0.0, 1.0)
                        scores = _central_metrics(display_render, target)
                        prediction_np = display_render.detach().cpu().numpy().astype(np.float32)
                        mechanisms = mechanism_metrics(prediction_np, image, masks)
                        components = codec.blob_components(blob)
                        stream_bytes = len(blob)
                        row = {
                            "schema": SCHEMA,
                            "status": "ok",
                            "manifest_sha256": manifest["manifest_sha256"],
                            "candidate_key": candidate_key,
                            "fit_key": key,
                            "image_id": image_id,
                            "arm": arm,
                            "n_gaussians": int(candidate_field.n),
                            "height": height,
                            "width": width,
                            "original_pixel_denominator": height * width,
                            "bit_mix": list(map(int, mix)),
                            "bits": "/".join(map(str, mix)),
                            "qat_iters": int(manifest["qat_iters"]),
                            "bytes": stream_bytes,
                            "bpp": actual_bpp(stream_bytes, height, width),
                            "stream_path": str(stream_path.relative_to(output)),
                            "stream_sha256": sha256_file(stream_path),
                            "previous_stream_sha256": previous_stream_sha256,
                            "stream_reencoded_identical": (
                                previous_stream_sha256 is None
                                or previous_stream_sha256 == sha256_file(stream_path)
                            ),
                            "stream_components": components,
                            "encode_seconds": encode_seconds,
                            "qat_seconds": qat_seconds,
                            "in_memory_decode_seconds": in_memory_decode_seconds,
                            "decode_seconds": decode_seconds,
                            "render_seconds": render_seconds,
                            "cold_decode_render_seconds": cold_decode_render_seconds,
                            "render_fps": 1.0 / max(render_seconds, 1e-12),
                            "cold_parity_max_abs": parity_max_abs,
                            "cold_parity_atol": parity_atol,
                            "cold_parity_domain": "persisted_stream_decoded_field_state",
                            "in_memory_decoded_field_sha256": reference_field_sha256,
                            "cold_decoded_field_sha256": cold_field_sha256,
                            "candidate_revalidated": bool(revalidate_candidates),
                            "fit_plus_search_seconds": float(fit_row["fit_plus_init_seconds"])
                            + qat_seconds + encode_seconds,
                            **scores,
                            **mechanisms,
                            "process_peak_rss_mb": _process_peak_rss_mb(),
                            **_gpu_peak_mb(torch, device),
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        }
                        _append_jsonl(candidate_journal, row)
                        existing_candidates[candidate_key] = row
                        completed_candidates += 1
                        if verbose:
                            print(
                                f"  stream {row['bits']} {row['bpp']:.4f} bpp "
                                f"{row['psnr']:.3f} dB ({stream_bytes} B)",
                                flush=True,
                            )
                        del (
                            in_memory_decoded_field,
                            cold_decoded_field,
                            cold_render,
                            display_render,
                        )
                    except Exception as exc:
                        failed = {
                            "schema": SCHEMA,
                            "status": "failed",
                            "manifest_sha256": manifest["manifest_sha256"],
                            "candidate_key": candidate_key,
                            "fit_key": key,
                            "image_id": image_id,
                            "arm": arm,
                            "n_gaussians": int(count),
                            "bit_mix": list(map(int, mix)),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                        }
                        _append_jsonl(candidate_journal, failed)
                        _append_jsonl(error_journal, failed)
                        existing_candidates[candidate_key] = failed
                        if verbose:
                            print(
                                f"FAILED stream {image_id} {arm} N={count} {mix}: {exc}",
                                flush=True,
                            )
                del field
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
    return {
        "status": "complete_for_requested_subset",
        "new_fits": new_fits,
        "completed_fits": completed_fits,
        "completed_candidates": completed_candidates,
        "plan": dry_run_plan(manifest),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {}
        for key, value in row.items():
            flat[key] = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else value
            )
        flattened.append(flat)
    fields = sorted({key for row in flattened for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(np.asarray(image) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", compress_level=9, optimize=False)


def _retain_stream(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise ValueError(f"retained stream collision at {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _responsibility_summary(
    field: Any,
    height: int,
    width: int,
    target_masks: dict[str, np.ndarray | float],
    header: dict[str, Any],
) -> dict[str, Any]:
    from structsplat.visualize import responsibility_diagnostics

    diagnostics = responsibility_diagnostics(
        field,
        height,
        width,
        sigma_cutoff=float(header.get("sigma_cutoff", 3.0)),
        support_fade=bool(header.get("support_fade", False)),
        chunk=int(header.get("render_chunk", 512)),
    )
    edge = np.asarray(target_masks["edge_band"], dtype=bool)
    texture = np.asarray(target_masks["texture_band"], dtype=bool)

    def masked_mean(array: np.ndarray, mask: np.ndarray) -> float | None:
        return float(array[mask].mean()) if mask.any() else None

    covered = diagnostics.denominator > 0.0
    return {
        "effective_count_mean": float(diagnostics.effective_count.mean()),
        "effective_count_edge_mean": masked_mean(diagnostics.effective_count, edge),
        "effective_count_texture_mean": masked_mean(diagnostics.effective_count, texture),
        "responsibility_entropy_mean": float(diagnostics.entropy.mean()),
        "responsibility_entropy_edge_mean": masked_mean(diagnostics.entropy, edge),
        "denominator_mean": float(diagnostics.denominator.mean()),
        "denominator_edge_mean": masked_mean(diagnostics.denominator, edge),
        "coverage_fraction": float(covered.mean()),
        "active_count_mean": float(diagnostics.active_count.mean()),
    }


def _enrich_selected_candidate(
    row: dict[str, Any],
    manifest: dict[str, Any],
    data_root: Path,
    outdir: Path,
    image_entry: dict[str, Any],
    target_bpp: float,
    device: str,
) -> dict[str, Any]:
    import torch
    from structsplat import codec, metrics

    stream_path = outdir / row["stream_path"]
    if sha256_file(stream_path) != row["stream_sha256"]:
        raise ValueError(f"selected stream hash mismatch: {stream_path}")
    blob = stream_path.read_bytes()
    header = codec.blob_header(blob)
    field = codec.decode(blob, device=device)
    prediction = _render_decoded_field(field, header).clamp(0.0, 1.0)
    source = load_native_rgb(data_root / image_entry["path"])
    target = torch.as_tensor(source, dtype=torch.float32, device=device)
    masks = mechanism_masks(source)
    enriched = dict(row)
    enriched.update(_central_metrics(prediction, target))
    enriched.update(mechanism_metrics(prediction.detach().cpu().numpy(), source, masks))
    enriched.update(_responsibility_summary(
        field, image_entry["height"], image_entry["width"], masks, header
    ))
    enriched["lpips"] = None
    enriched["lpips_error"] = None
    if manifest["central_scoring"].get("lpips_on_selected", False):
        try:
            enriched["lpips"] = metrics.LPIPS.distance(prediction, target)
        except Exception as exc:  # optional dependency/model cache must not erase other evidence
            enriched["lpips_error"] = f"{type(exc).__name__}: {exc}"
    target_label = f"{float(target_bpp):g}".replace(".", "p")
    selected_stream = (
        outdir / "streams" / "selected" / row["image_id"] / row["arm"]
        / f"target-{target_label}-{row['candidate_key']}.sspl1"
    )
    _retain_stream(stream_path, selected_stream)
    reconstruction = (
        outdir / "reconstructions" / row["image_id"] / row["arm"]
        / f"target-{target_label}-{row['candidate_key']}.png"
    )
    _save_rgb(reconstruction, prediction.detach().cpu().numpy())
    enriched.update({
        "target_bpp": float(target_bpp),
        "byte_cap": exact_byte_cap(target_bpp, image_entry["height"], image_entry["width"]),
        "selected_stream_path": str(selected_stream.relative_to(outdir)),
        "reconstruction_path": str(reconstruction.relative_to(outdir)),
        "selection_status": "ok",
    })
    return enriched


def _paired_values(
    selected: dict[tuple[str, str, float], dict[str, Any]],
    candidate_arm: str,
    control_arm: str,
    target: float,
    metric: str,
    *,
    sign: float = 1.0,
) -> list[float]:
    image_ids = sorted({key[0] for key in selected})
    values: list[float] = []
    for image_id in image_ids:
        candidate = selected.get((image_id, candidate_arm, float(target)))
        control = selected.get((image_id, control_arm, float(target)))
        if not candidate or not control:
            continue
        if candidate.get(metric) is None or control.get(metric) is None:
            continue
        values.append(sign * (float(candidate[metric]) - float(control[metric])))
    return values


def _resource_totals(
    fit_rows: Sequence[dict[str, Any]], candidate_rows: Sequence[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, float]]:
    totals: dict[tuple[str, str], dict[str, float]] = {}
    for row in fit_rows:
        if row.get("status") != "ok":
            continue
        key = (row["image_id"], row["arm"])
        bucket = totals.setdefault(key, {"fit_plus_init_seconds": 0.0, "codec_search_seconds": 0.0})
        bucket["fit_plus_init_seconds"] += float(row["fit_plus_init_seconds"])
    for row in candidate_rows:
        if row.get("status") != "ok":
            continue
        key = (row["image_id"], row["arm"])
        bucket = totals.setdefault(key, {"fit_plus_init_seconds": 0.0, "codec_search_seconds": 0.0})
        bucket["codec_search_seconds"] += float(row.get("qat_seconds", 0.0)) + float(
            row.get("encode_seconds", 0.0)
        )
    for bucket in totals.values():
        bucket["fit_plus_search_seconds"] = (
            bucket["fit_plus_init_seconds"] + bucket["codec_search_seconds"]
        )
    return totals


def _comparison_summary(
    manifest: dict[str, Any],
    selected_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    fit_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    selected = {
        (row["image_id"], row["arm"], float(row["target_bpp"])): row
        for row in selected_rows if row.get("selection_status") == "ok"
    }
    curves: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        if row.get("status") == "ok":
            curves.setdefault((row["image_id"], row["arm"]), []).append(row)
    resources = _resource_totals(fit_rows, candidate_rows)
    controls = [arm for arm in DIRECT_CONTROL_ARMS if arm in _arm_names(manifest)]
    comparisons: dict[str, Any] = {}
    p_values: dict[str, float | None] = {}
    for control in controls:
        target_stats: dict[str, Any] = {}
        for target in manifest["targets_bpp"]:
            deltas = _paired_values(
                selected, PRIMARY_ARM, control, float(target), "psnr"
            )
            stats = cluster_bootstrap(deltas)
            target_stats[f"{float(target):g}"] = stats
            p_values[f"{control}:psnr:{float(target):g}"] = stats["p_two_sided"]
        per_image_bd: list[dict[str, Any]] = []
        for image in manifest["images"]:
            image_id = image["id"]
            result = bd_rate(
                curves.get((image_id, PRIMARY_ARM), []),
                curves.get((image_id, control), []),
            )
            per_image_bd.append({"image_id": image_id, **result})
        bd_values = [
            float(row["bd_rate_percent"]) for row in per_image_bd if row.get("defined")
        ]
        bd_stats = cluster_bootstrap(bd_values)
        p_values[f"{control}:bd_rate"] = bd_stats["p_two_sided"]

        time_ratios: list[float] = []
        for image in manifest["images"]:
            candidate_time = resources.get((image["id"], PRIMARY_ARM), {}).get(
                "fit_plus_search_seconds"
            )
            control_time = resources.get((image["id"], control), {}).get(
                "fit_plus_search_seconds"
            )
            if candidate_time is not None and control_time and control_time > 0.0:
                time_ratios.append(candidate_time / control_time)
        edge_deltas: list[float] = []
        texture_relative: list[float] = []
        bleed_deltas: list[float] = []
        for target in manifest["targets_bpp"]:
            edge_deltas.extend(_paired_values(
                selected, PRIMARY_ARM, control, float(target), "edge_mse"
            ))
            bleed_deltas.extend(_paired_values(
                selected, PRIMARY_ARM, control, float(target), "signed_cross_edge_bleed"
            ))
            image_ids = [image["id"] for image in manifest["images"]]
            for image_id in image_ids:
                candidate = selected.get((image_id, PRIMARY_ARM, float(target)))
                reference = selected.get((image_id, control, float(target)))
                if not candidate or not reference or not reference.get("texture_mse"):
                    continue
                texture_relative.append(
                    (float(candidate["texture_mse"]) - float(reference["texture_mse"]))
                    / float(reference["texture_mse"])
                )
        comparisons[control] = {
            "label": ARMS[control]["label"],
            "target_psnr_delta_db": target_stats,
            "per_image_bd_rate": per_image_bd,
            "bd_rate_percent": bd_stats,
            "fit_plus_search_time_ratio": cluster_bootstrap(time_ratios),
            "edge_mse_delta": cluster_bootstrap(edge_deltas),
            "signed_cross_edge_bleed_delta": cluster_bootstrap(bleed_deltas),
            "texture_mse_relative_delta": cluster_bootstrap(texture_relative),
        }
    adjusted = holm_adjust(p_values)
    for control, comparison in comparisons.items():
        comparison["holm_adjusted_p"] = {
            "bd_rate": adjusted.get(f"{control}:bd_rate"),
            "target_psnr": {
                key: adjusted.get(f"{control}:psnr:{key}")
                for key in comparison["target_psnr_delta_db"]
            },
        }
    return {"comparisons": comparisons, "resource_totals": {
        f"{image}:{arm}": value for (image, arm), value in resources.items()
    }}


def evaluate_stage_gate(
    manifest: dict[str, Any],
    comparison_summary: dict[str, Any],
    *,
    search_complete: bool = True,
) -> dict[str, Any]:
    if manifest["stage"] != "stage1":
        return {
            "applicable": False,
            "pass": False,
            "reason": f"gate is defined only for Stage 1, not {manifest['stage']}",
        }
    if not search_complete:
        return {
            "applicable": True,
            "pass": False,
            "reason": "equal-search matrix is incomplete or contains failed candidates",
            "stage2_authorized": False,
        }
    comparisons = comparison_summary["comparisons"]
    complete_controls: list[str] = []
    for control, comparison in comparisons.items():
        stats = comparison["target_psnr_delta_db"]
        if all(stats.get(f"{float(target):g}", {}).get("n_images", 0) == len(manifest["images"])
               for target in manifest["targets_bpp"]):
            complete_controls.append(control)
    expected_controls = [arm for arm in DIRECT_CONTROL_ARMS if arm in _arm_names(manifest)]
    if set(complete_controls) != set(expected_controls):
        return {
            "applicable": True,
            "pass": False,
            "reason": "one or more preregistered direct controls are incomplete",
            "complete_direct_controls": complete_controls,
            "missing_direct_controls": sorted(set(expected_controls) - set(complete_controls)),
            "stage2_authorized": False,
        }
    # "Strongest" is frozen as the direct control with the smallest mean tensor-minus-control
    # PSNR delta over the two Stage-1 targets (highest control PSNR on common cells).
    strongest = min(
        complete_controls,
        key=lambda control: np.mean([
            comparisons[control]["target_psnr_delta_db"][f"{float(target):g}"]["mean"]
            for target in manifest["targets_bpp"]
        ]),
    )
    comparison = comparisons[strongest]
    bd = comparison["bd_rate_percent"]
    bd_condition = bool(
        bd.get("n_images", 0) == len(manifest["images"])
        and bd.get("mean") is not None
        and float(bd["mean"]) <= -10.0
        and bd["ci95"][1] is not None
        and float(bd["ci95"][1]) < 0.0
    )
    target_condition = True
    target_details: dict[str, bool] = {}
    for target in (0.5, 1.0):
        stats = comparison["target_psnr_delta_db"].get(f"{target:g}", {})
        passed = bool(
            stats.get("mean") is not None
            and float(stats["mean"]) >= 0.25
            and stats.get("ci95", [None])[0] is not None
            and float(stats["ci95"][0]) > 0.0
        )
        target_details[f"{target:g}"] = passed
        target_condition &= passed
    time_stats = comparison["fit_plus_search_time_ratio"]
    time_condition = bool(
        time_stats.get("mean") is not None and float(time_stats["mean"]) <= 1.10
    )
    edge = comparison["edge_mse_delta"].get("mean")
    bleed = comparison["signed_cross_edge_bleed_delta"].get("mean")
    texture = comparison["texture_mse_relative_delta"].get("mean")
    mechanism_condition = bool(
        ((edge is not None and float(edge) < 0.0) or (bleed is not None and float(bleed) < 0.0))
        and texture is not None
        and float(texture) <= float(manifest["mechanism_protocol"]["texture_failure_relative_threshold"])
    )
    passed = bool((bd_condition or target_condition) and time_condition and mechanism_condition)
    return {
        "applicable": True,
        "pass": passed,
        "strongest_direct_control": strongest,
        "quality_condition": {
            "bd_rate": bd_condition,
            "target_psnr_both": target_condition,
            "target_details": target_details,
        },
        "time_condition": time_condition,
        "mechanism_condition": mechanism_condition,
        "stage2_authorized": passed,
        "interpretation": (
            "Stage 1 authorizes Stage 2 but is not held-out evidence" if passed else
            "Stop: do not launch or interpret Stage 2 as a rescue run"
        ),
    }


def analyze_manifest(
    manifest_path: str | Path,
    data_root: str | Path,
    outdir: str | Path,
    *,
    device: str | None = None,
    make_figures: bool = True,
) -> dict[str, Any]:
    """Apply exact-cap RDO, selected diagnostics, inference, gate, and figures."""
    import torch

    manifest = load_manifest(manifest_path)
    validate_manifest_sources(manifest, data_root)
    output = Path(outdir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fit_rows_all = read_jsonl(output / "journals" / "fits.jsonl")
    candidate_rows_all = read_jsonl(output / "journals" / "candidates.jsonl")
    fit_latest = {
        row["fit_key"]: row for row in fit_rows_all
        if row.get("manifest_sha256") == manifest["manifest_sha256"]
    }
    candidate_latest = {
        row["candidate_key"]: row for row in candidate_rows_all
        if row.get("manifest_sha256") == manifest["manifest_sha256"]
    }
    fit_rows = list(fit_latest.values())
    candidate_rows = list(candidate_latest.values())

    expected_candidates: list[dict[str, Any]] = []
    for image in manifest["images"]:
        for arm in _arm_names(manifest):
            for count in image["candidate_counts"]:
                fit_key = fit_cell_key(manifest["manifest_sha256"], image["id"], arm, count)
                for mix in manifest["bit_mixes"]:
                    key = candidate_cell_key(fit_key, mix, manifest["qat_iters"])
                    row = candidate_latest.get(key)
                    if row is None or row.get("status") != "ok":
                        expected_candidates.append({
                            "candidate_key": key,
                            "image_id": image["id"],
                            "arm": arm,
                            "n_gaussians": count,
                            "bit_mix": mix,
                            "status": "missing" if row is None else "failed",
                            "error": None if row is None else row.get("error"),
                        })

    selected_rows: list[dict[str, Any]] = []
    enriched_cache: dict[str, dict[str, Any]] = {}
    ok_candidates = [row for row in candidate_rows if row.get("status") == "ok"]
    for image in manifest["images"]:
        for arm in _arm_names(manifest):
            rows = [
                row for row in ok_candidates
                if row["image_id"] == image["id"] and row["arm"] == arm
            ]
            for target_bpp in manifest["targets_bpp"]:
                chosen = select_under_cap(
                    rows, float(target_bpp), image["height"], image["width"]
                )
                if chosen is None:
                    selected_rows.append({
                        "schema": SCHEMA,
                        "manifest_sha256": manifest["manifest_sha256"],
                        "image_id": image["id"],
                        "arm": arm,
                        "target_bpp": float(target_bpp),
                        "byte_cap": exact_byte_cap(
                            float(target_bpp), image["height"], image["width"]
                        ),
                        "selection_status": "missing",
                        "reason": "no_measured_candidate_under_exact_byte_cap",
                    })
                    continue
                cached = enriched_cache.get(chosen["candidate_key"])
                if cached is None:
                    cached = _enrich_selected_candidate(
                        chosen,
                        manifest,
                        Path(data_root),
                        output,
                        image,
                        float(target_bpp),
                        device,
                    )
                    enriched_cache[chosen["candidate_key"]] = cached
                selected = dict(cached)
                # The same stream may be selected for multiple caps; keep cap-specific metadata.
                selected["target_bpp"] = float(target_bpp)
                selected["byte_cap"] = exact_byte_cap(
                    float(target_bpp), image["height"], image["width"]
                )
                target_label = f"{float(target_bpp):g}".replace(".", "p")
                selected_stream = (
                    output / "streams" / "selected" / chosen["image_id"] / chosen["arm"]
                    / f"target-{target_label}-{chosen['candidate_key']}.sspl1"
                )
                _retain_stream(output / chosen["stream_path"], selected_stream)
                reconstruction = (
                    output / "reconstructions" / chosen["image_id"] / chosen["arm"]
                    / f"target-{target_label}-{chosen['candidate_key']}.png"
                )
                _retain_stream(output / cached["reconstruction_path"], reconstruction)
                selected["selected_stream_path"] = str(selected_stream.relative_to(output))
                selected["reconstruction_path"] = str(reconstruction.relative_to(output))
                selected_rows.append(selected)

    completion = {
        "expected_fits": dry_run_plan(manifest)["fits"],
        "successful_fits": sum(row.get("status") == "ok" for row in fit_rows),
        "failed_fits": sum(row.get("status") == "failed" for row in fit_rows),
        "expected_candidates": dry_run_plan(manifest)["codec_candidates"],
        "successful_candidates": len(ok_candidates),
        "missing_or_failed_candidates": len(expected_candidates),
        "complete": len(expected_candidates) == 0,
    }
    comparison = _comparison_summary(manifest, selected_rows, candidate_rows, fit_rows)
    gate = evaluate_stage_gate(
        manifest, comparison, search_complete=bool(completion["complete"])
    )
    analysis = {
        "schema": SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "stage": manifest["stage"],
        "scientific_scope": manifest["scientific_scope"],
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "completion": completion,
        "missing_or_failed_candidates": expected_candidates,
        **comparison,
        "gate": gate,
        "interpretation_guard": (
            "Stage-0 output validates plumbing only and cannot support a research conclusion"
            if manifest["stage"] == "stage0a" else
            "Stage-1 output is a killing gate, not positive held-out evidence"
            if manifest["stage"] == "stage1" else
            "Stage-2 is held-out confirmation only if authorized by the frozen Stage-1 gate"
        ),
    }
    _atomic_json(output / "analysis" / "summary.json", analysis)
    _atomic_json(output / "analysis" / "selected.json", selected_rows)
    _atomic_json(output / "analysis" / "missing.json", expected_candidates)
    _write_csv(output / "analysis" / "raw_candidates.csv", candidate_rows)
    _write_csv(output / "analysis" / "selected.csv", selected_rows)
    if make_figures:
        generate_publication_figures(
            manifest, Path(data_root), output, candidate_rows, selected_rows, analysis,
            device=device,
        )
    generate_html_index(manifest, output, selected_rows, analysis)
    return analysis


def sweep_conventional_codecs(
    manifest_path: str | Path,
    data_root: str | Path,
    outdir: str | Path,
    *,
    jpeg_qualities: Sequence[int] = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95),
    avif_qualities: Sequence[int] = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95),
    require_modern: bool = True,
) -> dict[str, Any]:
    """Sweep PNG, JPEG 4:4:4, and AVIF 4:4:4 to actual decoder-complete bytes.

    These rows are outside-class context only and never enter the tensor-WSE gate.
    Metadata is omitted, native dimensions are retained, and every output is decoded
    centrally to RGB before scoring.
    """
    import torch
    from structsplat import metrics

    manifest = load_manifest(manifest_path)
    validate_manifest_sources(manifest, data_root)
    output = Path(outdir)
    try:
        import pillow_avif  # noqa: F401
        avif_available = True
        avif_version = getattr(pillow_avif, "__version__", None)
    except ImportError:
        avif_available = False
        avif_version = None
    if require_modern and not avif_available:
        raise RuntimeError(
            "AVIF context is required but pillow-avif-plugin is unavailable; "
            "install the benchmark extra with pip install -e '.[benchmark]'"
        )
    rows: list[dict[str, Any]] = []
    for entry in manifest["images"]:
        source_path = Path(data_root) / entry["path"]
        target_np = load_native_rgb(source_path)
        target = torch.as_tensor(target_np, dtype=torch.float32)
        source_pil = Image.open(source_path).convert("RGB")
        recipes: list[tuple[str, int | None, dict[str, Any], str]] = [
            ("png", None, {"compress_level": 9, "optimize": False}, "PNG")
        ]
        recipes.extend(
            ("jpeg_444", int(quality), {
                "quality": int(quality), "subsampling": 0, "optimize": True, "progressive": False,
            }, "JPEG")
            for quality in jpeg_qualities
        )
        if avif_available:
            recipes.extend(
                ("avif_444", int(quality), {
                    "quality": int(quality), "subsampling": "4:4:4", "speed": 6,
                }, "AVIF")
                for quality in avif_qualities
            )
        for codec_name, quality, kwargs, format_name in recipes:
            stream = io.BytesIO()
            start = time.perf_counter()
            source_pil.save(stream, format=format_name, **kwargs)
            payload = stream.getvalue()
            encode_seconds = time.perf_counter() - start
            extension = "jpg" if format_name == "JPEG" else format_name.lower()
            quality_label = "lossless" if quality is None else f"q{quality:03d}"
            path = output / "conventional" / entry["id"] / f"{codec_name}-{quality_label}.{extension}"
            _atomic_bytes(path, payload)
            decode_start = time.perf_counter()
            decoded = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.float32) / 255.0
            decode_seconds = time.perf_counter() - decode_start
            prediction = torch.as_tensor(decoded, dtype=torch.float32)
            rows.append({
                "schema": SCHEMA,
                "manifest_sha256": manifest["manifest_sha256"],
                "context_only": True,
                "image_id": entry["id"],
                "codec": codec_name,
                "quality": quality,
                "height": entry["height"],
                "width": entry["width"],
                "bytes": len(payload),
                "bpp": actual_bpp(len(payload), entry["height"], entry["width"]),
                "psnr": float(metrics.psnr(prediction, target)),
                "ssim": float(metrics.ssim(prediction, target)),
                "ms_ssim": float(metrics.ms_ssim(prediction, target)),
                "encode_seconds": encode_seconds,
                "decode_seconds": decode_seconds,
                "stream_path": str(path.relative_to(output)),
                "stream_sha256": sha256_file(path),
                "metadata_policy": "none",
                "central_round_trip": "Pillow RGB",
                "chroma_mode": "4:4:4" if codec_name != "png" else "RGB lossless",
            })
    payload = {
        "schema": SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "context_only": True,
        "encoder_versions": {
            "pillow": environment_provenance()["pillow"],
            "pillow_avif_plugin": avif_version,
        },
        "modern_codec_available": avif_available,
        "rows": rows,
    }
    _atomic_json(output / "analysis" / "conventional.json", payload)
    _write_csv(output / "analysis" / "conventional.csv", rows)
    return payload


def calibrate_bytes_per_gaussian(
    image_paths: Sequence[str | Path],
    data_root: str | Path,
    outdir: str | Path,
    *,
    count_densities_per_megapixel: Sequence[float] = (2048.0, 8192.0),
    iters: int = 50,
    bit_mix: Sequence[int] = (16, 8, 8, 8),
    device: str | None = None,
    renderer: str = "normalized",
    render_chunk: int = 512,
) -> dict[str, Any]:
    """Stage-0b-only rate calibration; it never compares arm quality."""
    import torch
    from structsplat import codec
    from structsplat import init as init_ops
    from structsplat.config import FitConfig, InitConfig
    from structsplat.fit import fit

    root = Path(data_root)
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if renderer not in BENCHMARK_RENDERERS:
        raise ValueError(
            f"BENCH-007 renderer must be one of {BENCHMARK_RENDERERS}, got {renderer!r}"
        )
    if renderer == "cuda" and not str(device).startswith("cuda"):
        raise ValueError("Stage-0b renderer='cuda' requires a CUDA device")
    entries = []
    for raw_path in sorted({str(Path(path)) for path in image_paths}):
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        rgb_hash, height, width = decoded_rgb_sha256(path)
        entries.append({
            "id": _infer_image_id(path),
            "path": _relative_to_root(path, root),
            "source_sha256": sha256_file(path),
            "decoded_rgb_sha256": rgb_hash,
            "height": height,
            "width": width,
        })
    if set(entry["id"] for entry in entries) != set(STAGE_IDS["stage0b"]):
        raise ValueError(
            f"Stage 0b requires IDs {list(STAGE_IDS['stage0b'])}; "
            f"got {[entry['id'] for entry in entries]}"
        )
    config = {
        "schema": SCHEMA,
        "stage": "stage0b",
        "scientific_scope": "rate_calibration_only_no_method_quality_comparison",
        "images": entries,
        "count_densities_per_megapixel": [float(value) for value in count_densities_per_megapixel],
        "strategy": "quadtree_wse",
        "iters": int(iters),
        "bit_mix": list(map(int, bit_mix)),
        "seed": 0,
        "renderer": renderer,
        "renderer_protocol": {
            "equation": "normalized_weighted_sum",
            "implementation": (
                "pytorch_reference" if renderer == "normalized" else "owned_exact_cuda"
            ),
            "cuda_is_equation_preserving": renderer == "cuda",
        },
        "render_chunk": int(render_chunk),
        "repository": repository_provenance(),
        "environment": environment_provenance(),
    }
    config["calibration_sha256"] = _sha256_bytes(_canonical_json(config))
    _atomic_json(output / "calibration_config.json", config)
    journal = output / "calibration.jsonl"
    latest = _latest_rows(journal, "calibration_key")
    rows: list[dict[str, Any]] = []
    for entry in entries:
        image = load_native_rgb(root / entry["path"])
        target = torch.as_tensor(image, dtype=torch.float32, device=device)
        for density in count_densities_per_megapixel:
            count = max(64, int(math.floor(float(density) * entry["height"] * entry["width"] / 1e6 + 0.5)))
            key = _sha256_bytes(_canonical_json([
                config["calibration_sha256"], entry["id"], count
            ]))[:24]
            previous = latest.get(key)
            if previous and previous.get("status") == "ok":
                rows.append(previous)
                continue
            start = time.perf_counter()
            field = init_ops.build_field(
                image,
                InitConfig(strategy="quadtree_wse", num_gaussians=count, seed=0),
                device=device,
            )
            init_seconds = time.perf_counter() - start
            fcfg = FitConfig(
                iters=int(iters),
                renderer=renderer,
                render_chunk=int(render_chunk),
                checkpoint_policy="best_psnr_final_count",
            )
            fit_start = time.perf_counter()
            result = fit(field, target, fcfg, verbose=False)
            _gpu_sync(torch, device)
            fit_seconds = time.perf_counter() - fit_start
            ccfg = codec.CodecConfig(
                bits_means=int(bit_mix[0]),
                bits_scales=int(bit_mix[1]),
                bits_rot=int(bit_mix[2]),
                bits_colors=int(bit_mix[3]),
            )
            encode_start = time.perf_counter()
            blob = codec.encode(result["field"], entry["height"], entry["width"], ccfg, fcfg)
            encode_seconds = time.perf_counter() - encode_start
            parts = codec.blob_components(blob)
            row = {
                "schema": SCHEMA,
                "status": "ok",
                "calibration_sha256": config["calibration_sha256"],
                "calibration_key": key,
                "image_id": entry["id"],
                "height": entry["height"],
                "width": entry["width"],
                "n_gaussians": int(result["field"].n),
                "count_density_per_megapixel": float(density),
                "bytes": len(blob),
                "bpp": actual_bpp(len(blob), entry["height"], entry["width"]),
                "bytes_per_gaussian": len(blob) / float(result["field"].n),
                "payload_bytes_per_gaussian": (
                    len(blob) - int(parts["header_bytes"])
                ) / float(result["field"].n),
                "header_bytes": int(parts["header_bytes"]),
                "stream_sha256": _sha256_bytes(blob),
                "init_seconds": init_seconds,
                "fit_seconds": fit_seconds,
                "encode_seconds": encode_seconds,
                "fit_psnr_diagnostic_only": float(result["psnr"]),
                "process_peak_rss_mb": _process_peak_rss_mb(),
                **_gpu_peak_mb(torch, device),
            }
            _append_jsonl(journal, row)
            rows.append(row)
            print(
                f"calibration {entry['id']} N={count}: "
                f"{row['bytes_per_gaussian']:.3f} B/G",
                flush=True,
            )
            del field
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    values = [float(row["bytes_per_gaussian"]) for row in rows]
    result_payload = {
        "schema": SCHEMA,
        "stage": "stage0b",
        "scientific_scope": "rate_calibration_only_no_method_quality_comparison",
        "calibration_sha256": config["calibration_sha256"],
        "rows": rows,
        "recommended_bytes_per_gaussian": float(statistics.median(values)),
        "range_bytes_per_gaussian": [float(min(values)), float(max(values))],
        "estimator": "median_complete_SSPL1_bytes_per_fitted_gaussian_across_images_and_counts",
    }
    _atomic_json(output / "calibration_summary.json", result_payload)
    _write_csv(output / "calibration.csv", rows)
    return result_payload


FIGURE_SPECS = {
    "f5_causal_allocation.png": "F5 — equal-count causal allocation comparison",
    "f6_actual_rate_phase_diagram.png": "F6 — actual-rate phase diagram and SSPL1 bytes",
    "f7_mechanism.png": "F7 — edge, texture, bleed, and contributor mechanism",
    "f8_resources.png": "F8 — optimization and encoder/decoder resources",
    "f9_qualitative_quantiles.png": "F9 — preregistered success/median/failure cases",
}


def _figure_scope_banner(figure: Any, manifest: dict[str, Any]) -> None:
    figure.text(
        0.5,
        0.008,
        f"{manifest['stage']} · {manifest['scientific_scope']} · actual SSPL1 bytes · "
        f"manifest {manifest['manifest_sha256'][:12]}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#475569",
    )


def _placeholder_figure(path: Path, title: str, reason: str, manifest: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
    axis.axis("off")
    axis.text(0.5, 0.64, title, ha="center", va="center", fontsize=17, weight="bold")
    axis.text(0.5, 0.38, reason, ha="center", va="center", fontsize=11, wrap=True)
    _figure_scope_banner(figure, manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _strongest_control(analysis: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    gate_control = analysis.get("gate", {}).get("strongest_direct_control")
    if gate_control:
        return str(gate_control)
    available = [arm for arm in DIRECT_CONTROL_ARMS if arm in _arm_names(manifest)]
    if "local_slic_sobel_control" in available:
        return "local_slic_sobel_control"
    return available[0] if available else None


def _load_stream_reconstruction(
    row: dict[str, Any], outdir: Path, *, device: str
) -> np.ndarray:
    from structsplat import codec

    blob = (outdir / row["stream_path"]).read_bytes()
    # Exact-CUDA stream semantics require CUDA tensors. Reuse the validated analysis device
    # instead of silently forcing the F5 replay onto CPU.
    prediction = codec.decode_and_render(blob, device=device).clamp(0.0, 1.0)
    return prediction.detach().cpu().numpy()


def _median_comparison_image(
    manifest: dict[str, Any], selected_rows: Sequence[dict[str, Any]], control: str | None
) -> str:
    if control is None:
        return manifest["images"][len(manifest["images"]) // 2]["id"]
    target = float(manifest["targets_bpp"][0])
    index = {
        (row["image_id"], row["arm"], float(row["target_bpp"])): row
        for row in selected_rows if row.get("selection_status") == "ok"
    }
    deltas: list[tuple[float, str]] = []
    for image in manifest["images"]:
        candidate = index.get((image["id"], PRIMARY_ARM, target))
        reference = index.get((image["id"], control, target))
        if candidate and reference:
            deltas.append((float(candidate["psnr"]) - float(reference["psnr"]), image["id"]))
    if not deltas:
        return manifest["images"][len(manifest["images"]) // 2]["id"]
    deltas.sort()
    return deltas[len(deltas) // 2][1]


def _plot_f5(
    manifest: dict[str, Any],
    data_root: Path,
    outdir: Path,
    candidate_rows: Sequence[dict[str, Any]],
    selected_rows: Sequence[dict[str, Any]],
    analysis: dict[str, Any],
    device: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from structsplat.gaussians import GaussianField

    control = _strongest_control(analysis, manifest)
    image_id = _median_comparison_image(manifest, selected_rows, control)
    image_entry = next(entry for entry in manifest["images"] if entry["id"] == image_id)
    image = load_native_rgb(data_root / image_entry["path"])
    counts = image_entry["candidate_counts"]
    count = counts[len(counts) // 2]
    mix = list(manifest["bit_mixes"][0])
    arms = _arm_names(manifest)
    fit_rows = _latest_rows(outdir / "journals" / "fits.jsonl", "fit_key")
    rows = {
        row["arm"]: row for row in candidate_rows
        if row.get("status") == "ok"
        and row["image_id"] == image_id
        and int(row["n_gaussians"]) == int(count)
        and list(row["bit_mix"]) == mix
    }
    if len(rows) < 2:
        _placeholder_figure(
            path, FIGURE_SPECS[path.name],
            f"Insufficient equal-count candidates for image {image_id}, N={count}, bits={mix}.",
            manifest,
        )
        return
    figure, axes = plt.subplots(
        2, len(arms) + 1, figsize=(3.0 * (len(arms) + 1), 6.1), constrained_layout=True
    )
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Target {image_id}")
    axes[1, 0].imshow(image)
    axes[1, 0].set_title("Native target")
    for axis in axes[:, 0]:
        axis.axis("off")
    for column, arm in enumerate(arms, 1):
        row = rows.get(arm)
        if row is None:
            axes[0, column].text(0.5, 0.5, "missing", ha="center", va="center")
            axes[1, column].text(0.5, 0.5, "missing", ha="center", va="center")
            axes[0, column].axis("off")
            axes[1, column].axis("off")
            continue
        fit_row = fit_rows[row["fit_key"]]
        initial = GaussianField.load(str(outdir / fit_row["initial_field_path"]), device="cpu")
        points = initial.means.detach().cpu().numpy()
        if len(points) > 1800:
            points = points[np.linspace(0, len(points) - 1, 1800).astype(int)]
        axes[0, column].imshow(image * 0.55)
        axes[0, column].scatter(
            points[:, 0], points[:, 1], s=2.0, c="#f8fafc", alpha=0.72, linewidths=0
        )
        axes[0, column].set_title(ARMS[arm]["label"], fontsize=10)
        reconstruction = _load_stream_reconstruction(row, outdir, device=device)
        axes[1, column].imshow(reconstruction)
        axes[1, column].set_title(
            f"{row['bpp']:.3f} bpp · {row['psnr']:.2f} dB", fontsize=9
        )
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.suptitle(
        f"F5. Same target, N={count:,}, bit mix {'/'.join(map(str, mix))}; "
        "top: source-only initial sites, bottom: cold-decoded output",
        fontsize=14,
    )
    _figure_scope_banner(figure, manifest)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _plot_f6(
    manifest: dict[str, Any],
    outdir: Path,
    candidate_rows: Sequence[dict[str, Any]],
    selected_rows: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    ok = [row for row in candidate_rows if row.get("status") == "ok"]
    selected = [row for row in selected_rows if row.get("selection_status") == "ok"]
    if not ok:
        _placeholder_figure(path, FIGURE_SPECS[path.name], "No successful codec candidates.", manifest)
        return
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for arm_index, arm in enumerate(_arm_names(manifest)):
        color = colors(arm_index % 10)
        per_image = []
        for image in manifest["images"]:
            envelope = monotone_envelope([
                row for row in ok if row["arm"] == arm and row["image_id"] == image["id"]
            ])
            if envelope:
                x = [row["bpp"] for row in envelope]
                y = [row["psnr"] for row in envelope]
                axes[0].plot(x, y, color=color, alpha=0.17, linewidth=0.8)
                per_image.append(envelope)
        target_x, target_y, lower, upper = [], [], [], []
        for target in manifest["targets_bpp"]:
            rows = [
                row for row in selected
                if row["arm"] == arm and float(row["target_bpp"]) == float(target)
            ]
            if not rows:
                continue
            values = [float(row["psnr"]) for row in rows]
            bootstrap = cluster_bootstrap(values)
            target_x.append(float(np.mean([row["bpp"] for row in rows])))
            target_y.append(float(bootstrap["mean"]))
            lower.append(float(bootstrap["mean"]) - float(bootstrap["ci95"][0]))
            upper.append(float(bootstrap["ci95"][1]) - float(bootstrap["mean"]))
        if target_x:
            axes[0].errorbar(
                target_x,
                target_y,
                yerr=np.asarray([lower, upper]),
                color=color,
                marker="o",
                linewidth=2,
                capsize=3,
                label=ARMS[arm]["label"],
            )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Actual self-contained SSPL1 rate (bpp)")
    axes[0].set_ylabel("Cold-decoded PSNR (dB)")
    axes[0].set_title("Measured per-image envelopes; image-cluster 95% intervals")
    axes[0].grid(True, which="both", alpha=0.22)
    axes[0].set_ylim(
        min(float(row["psnr"]) for row in ok) - 0.75,
        max(float(row["psnr"]) for row in ok) + 0.75,
    )
    conventional_path = outdir / "analysis" / "conventional.json"
    if conventional_path.exists():
        conventional = json.loads(conventional_path.read_text(encoding="utf-8"))["rows"]
        context_axis = axes[0].inset_axes([0.55, 0.49, 0.42, 0.39])
        context_values: list[float] = []
        for codec_name, marker, color in (
            ("jpeg_444", "s", "#111827"),
            ("avif_444", "D", "#7c3aed"),
        ):
            codec_rows = [row for row in conventional if row["codec"] == codec_name]
            qualities = sorted({row.get("quality") for row in codec_rows}, key=lambda value: -1 if value is None else value)
            points = []
            for quality in qualities:
                rows = [row for row in codec_rows if row.get("quality") == quality]
                if rows:
                    points.append((
                        float(np.mean([row["bpp"] for row in rows])),
                        float(np.mean([row["psnr"] for row in rows])),
                    ))
            points.sort()
            if points:
                context_values.extend(point[1] for point in points)
                context_axis.plot(
                    [point[0] for point in points], [point[1] for point in points],
                    linestyle="--", marker=marker, color=color, linewidth=1.2, markersize=4,
                    label=codec_name,
                )
        png_rows = [row for row in conventional if row["codec"] == "png"]
        if context_values:
            context_axis.set_ylim(min(context_values) - 1.0, max(context_values) + 2.5)
        if png_rows:
            png_x = float(np.mean([row["bpp"] for row in png_rows]))
            png_psnr = float(np.mean([row["psnr"] for row in png_rows]))
            context_axis.text(
                0.98,
                0.97,
                f"PNG lossless: {png_x:.1f} bpp\nPSNR capped at {png_psnr:.0f} dB (off-axis)",
                transform=context_axis.transAxes,
                fontsize=5.7,
                ha="right",
                va="top",
                color="#475569",
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1", "pad": 1.5},
            )
        context_axis.set_xscale("log", base=2)
        context_axis.set_title("Conventional context only", fontsize=7)
        context_axis.set_xlabel("actual bpp", fontsize=6)
        context_axis.set_ylabel("PSNR", fontsize=6)
        context_axis.tick_params(labelsize=6)
        context_axis.grid(True, alpha=0.18)
        context_axis.legend(fontsize=6, loc="lower right")
    missing_rows = [row for row in selected_rows if row.get("selection_status") != "ok"]
    if missing_rows:
        missing_summary = []
        for arm in _arm_names(manifest):
            for target in manifest["targets_bpp"]:
                count = sum(
                    row.get("arm") == arm and float(row.get("target_bpp", -1)) == float(target)
                    for row in missing_rows
                )
                if count:
                    missing_summary.append(f"{arm}@{float(target):g}: {count} missing")
        axes[0].text(
            0.01, 0.01, "\n".join(missing_summary), transform=axes[0].transAxes,
            va="bottom", ha="left", fontsize=7, color="#b91c1c",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#fecaca"},
        )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, legend_labels, fontsize=8)

    representative_target = min(
        manifest["targets_bpp"], key=lambda value: abs(float(value) - 1.0)
    )
    components = ["header", "means", "scales", "rotation", "colors", "opacity"]
    component_colors = ["#64748b", "#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#9333ea"]
    bottoms = np.zeros(len(_arm_names(manifest)))
    for component, color in zip(components, component_colors):
        values = []
        for arm in _arm_names(manifest):
            rows = [
                row for row in selected
                if row["arm"] == arm
                and float(row["target_bpp"]) == float(representative_target)
            ]
            per_row = []
            for row in rows:
                breakdown = row["stream_components"]
                if component == "header":
                    per_row.append(float(breakdown["header_bytes"]))
                elif component in breakdown["streams"]:
                    per_row.append(float(breakdown["streams"][component]["framed_bytes"]))
            values.append(float(np.mean(per_row)) if per_row else 0.0)
        axes[1].bar(
            np.arange(len(values)), values, bottom=bottoms, color=color, label=component
        )
        bottoms += np.asarray(values)
    axes[1].set_xticks(np.arange(len(_arm_names(manifest))))
    axes[1].set_xticklabels(
        [ARMS[arm]["label"] for arm in _arm_names(manifest)], rotation=28, ha="right", fontsize=8
    )
    axes[1].set_ylabel("Mean complete-stream bytes")
    axes[1].set_title(f"Exact SSPL1 byte components at {float(representative_target):g} bpp cap")
    axes[1].legend(fontsize=8, ncol=2)
    figure.suptitle("F6. Actual-rate structure phase diagram", fontsize=15)
    _figure_scope_banner(figure, manifest)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _plot_f7(
    manifest: dict[str, Any],
    selected_rows: Sequence[dict[str, Any]],
    analysis: dict[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    control = _strongest_control(analysis, manifest)
    if control is None:
        _placeholder_figure(path, FIGURE_SPECS[path.name], "No direct control is available.", manifest)
        return
    selected = {
        (row["image_id"], row["arm"], float(row["target_bpp"])): row
        for row in selected_rows if row.get("selection_status") == "ok"
    }
    metrics = [
        ("edge_mse", "Edge-band MSE", 1e3),
        ("texture_mse", "Texture-band MSE", 1e3),
        ("signed_cross_edge_bleed", "Signed bleed", 1e2),
        ("effective_count_edge_mean", "Effective contributors (edge)", 1.0),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(15.5, 4.7))
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.16, top=0.82, wspace=0.26)
    any_values = False
    for axis, (metric, label, scale) in zip(axes, metrics):
        positions, labels = [], []
        for target_index, target in enumerate(manifest["targets_bpp"]):
            values = []
            for image in manifest["images"]:
                candidate = selected.get((image["id"], PRIMARY_ARM, float(target)))
                reference = selected.get((image["id"], control, float(target)))
                if not candidate or not reference:
                    continue
                if candidate.get(metric) is None or reference.get(metric) is None:
                    continue
                values.append(scale * (float(candidate[metric]) - float(reference[metric])))
            if not values:
                continue
            any_values = True
            x = target_index
            jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else [0.0]
            axis.scatter(np.asarray(jitter) + x, values, s=24, alpha=0.58, color="#334155")
            stats = cluster_bootstrap(values)
            axis.errorbar(
                [x], [stats["mean"]],
                yerr=[[stats["mean"] - stats["ci95"][0]], [stats["ci95"][1] - stats["mean"]]],
                fmt="o", color="#dc2626", capsize=4, markersize=6,
            )
            positions.append(x)
            labels.append(f"{float(target):g}")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Target bpp")
        suffix = " ×10³" if scale == 1e3 else " ×10²" if scale == 1e2 else ""
        axis.set_ylabel(f"Tensor − control{suffix}")
        axis.set_title(label)
        axis.grid(True, axis="y", alpha=0.2)
    if not any_values:
        plt.close(figure)
        _placeholder_figure(path, FIGURE_SPECS[path.name], "No matched selected mechanism rows.", manifest)
        return
    figure.suptitle(
        f"F7. Mechanism deltas versus {ARMS[control]['label']} (dots: images; red: cluster mean/95%)",
        fontsize=14, y=0.96,
    )
    _figure_scope_banner(figure, manifest)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _plot_f8(
    manifest: dict[str, Any],
    outdir: Path,
    candidate_rows: Sequence[dict[str, Any]],
    selected_rows: Sequence[dict[str, Any]],
    analysis: dict[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    resource_rows = analysis.get("resource_totals", {})
    if not resource_rows:
        _placeholder_figure(path, FIGURE_SPECS[path.name], "No resource telemetry is available.", manifest)
        return
    arms = _arm_names(manifest)
    fit_values, search_values, decode_values, render_values = [], [], [], []
    for arm in arms:
        buckets = [
            value for key, value in resource_rows.items() if key.endswith(f":{arm}")
        ]
        fit_values.append(float(np.mean([row["fit_plus_init_seconds"] for row in buckets])))
        search_values.append(float(np.mean([row["codec_search_seconds"] for row in buckets])))
        candidates = [row for row in candidate_rows if row.get("status") == "ok" and row["arm"] == arm]
        decode_values.append(float(np.mean([row["decode_seconds"] for row in candidates])) if candidates else 0.0)
        render_values.append(float(np.mean([row["render_seconds"] for row in candidates])) if candidates else 0.0)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.1), constrained_layout=True)
    x = np.arange(len(arms))
    axes[0].bar(x, fit_values, label="init + fit", color="#2563eb")
    axes[0].bar(x, search_values, bottom=fit_values, label="QAT + codec search", color="#f59e0b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([ARMS[arm]["label"] for arm in arms], rotation=28, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean encoder seconds per image")
    axes[0].set_title("Full equal candidate search is charged")
    axes[0].legend(fontsize=8, loc="upper left")

    image_id = _median_comparison_image(manifest, selected_rows, _strongest_control(analysis, manifest))
    image_entry = next(entry for entry in manifest["images"] if entry["id"] == image_id)
    count = image_entry["candidate_counts"][len(image_entry["candidate_counts"]) // 2]
    fit_rows = [
        row for row in read_jsonl(outdir / "journals" / "fits.jsonl")
        if row.get("status") == "ok" and row["image_id"] == image_id
        and int(row["n_gaussians"]) == int(count)
    ]
    for arm_index, arm in enumerate(arms):
        row = next((value for value in fit_rows if value["arm"] == arm), None)
        if row is None:
            continue
        history = json.loads((outdir / row["history_path"]).read_text(encoding="utf-8"))
        if history.get("iter") and history.get("psnr"):
            axes[1].plot(
                history["iter"], history["psnr"], label=ARMS[arm]["label"],
                color=plt.get_cmap("tab10")(arm_index % 10),
            )
    axes[1].set_xlabel("Optimizer iteration")
    axes[1].set_ylabel("Fit PSNR (raw render, dB)")
    axes[1].set_title(f"Equal-horizon trajectories: image {image_id}, N={count:,}")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(fontsize=8)
    inset = axes[0].inset_axes([0.55, 0.52, 0.42, 0.42])
    width = 0.38
    inset.bar(x - width / 2, np.asarray(decode_values) * 1e3, width, label="decode")
    inset.bar(x + width / 2, np.asarray(render_values) * 1e3, width, label="render")
    inset.set_title("Mean decoder ms", fontsize=8)
    inset.set_xticks([])
    inset.legend(fontsize=6)
    figure.suptitle("F8. Optimization and resource accounting", fontsize=15)
    _figure_scope_banner(figure, manifest)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _edge_crop(image: np.ndarray, size: int = 256) -> tuple[slice, slice]:
    height, width = image.shape[:2]
    crop_h, crop_w = min(size, height), min(size, width)
    luma = image @ np.array([0.2126, 0.7152, 0.0722])
    gradient_x, gradient_y = _sobel_xy(luma)
    magnitude = np.hypot(gradient_x, gradient_y)
    # A coarse integral search selects the most edge-rich crop without looking at method errors.
    integral = np.pad(magnitude, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    step = max(1, min(crop_h, crop_w) // 8)
    best = (-math.inf, 0, 0)
    for y in range(0, height - crop_h + 1, step):
        for x in range(0, width - crop_w + 1, step):
            score = (
                integral[y + crop_h, x + crop_w] - integral[y, x + crop_w]
                - integral[y + crop_h, x] + integral[y, x]
            )
            if score > best[0]:
                best = (float(score), y, x)
    _, y, x = best
    return slice(y, y + crop_h), slice(x, x + crop_w)


def _plot_f9(
    manifest: dict[str, Any],
    data_root: Path,
    outdir: Path,
    selected_rows: Sequence[dict[str, Any]],
    analysis: dict[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    control = _strongest_control(analysis, manifest)
    target_rate = float(manifest["targets_bpp"][0])
    index = {
        (row["image_id"], row["arm"], float(row["target_bpp"])): row
        for row in selected_rows if row.get("selection_status") == "ok"
    }
    cases: list[tuple[float, str]] = []
    for image in manifest["images"]:
        candidate = index.get((image["id"], PRIMARY_ARM, target_rate))
        reference = index.get((image["id"], control, target_rate)) if control else None
        if candidate and reference:
            cases.append((float(candidate["psnr"]) - float(reference["psnr"]), image["id"]))
    if not cases:
        _placeholder_figure(path, FIGURE_SPECS[path.name], "No matched qualitative cases.", manifest)
        return
    cases.sort()
    chosen = [cases[0], cases[len(cases) // 2], cases[-1]]
    labels = ["Failure", "Median", "Success"]
    rendered: list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[slice, slice]]] = []
    all_errors: list[np.ndarray] = []
    for _, image_id in chosen:
        entry = next(image for image in manifest["images"] if image["id"] == image_id)
        target = load_native_rgb(data_root / entry["path"])
        reference = np.asarray(Image.open(outdir / index[(image_id, control, target_rate)]["reconstruction_path"]).convert("RGB"), dtype=np.float32) / 255.0
        candidate = np.asarray(Image.open(outdir / index[(image_id, PRIMARY_ARM, target_rate)]["reconstruction_path"]).convert("RGB"), dtype=np.float32) / 255.0
        crop = _edge_crop(target)
        error_reference = np.abs(reference - target).mean(axis=2)
        error_candidate = np.abs(candidate - target).mean(axis=2)
        all_errors.extend([error_reference[crop], error_candidate[crop]])
        rendered.append((target, reference, candidate, crop))
    vmax = max(float(np.quantile(np.concatenate([value.ravel() for value in all_errors]), 0.99)), 1e-4)
    figure, axes = plt.subplots(3, 5, figsize=(15, 9.0))
    figure.subplots_adjust(left=0.02, right=0.995, bottom=0.07, top=0.90, wspace=0.03, hspace=0.04)
    for row_index, ((delta, image_id), label, values) in enumerate(zip(chosen, labels, rendered)):
        target, reference, candidate, crop = values
        panels = [target[crop], reference[crop], candidate[crop]]
        for column, panel in enumerate(panels):
            axes[row_index, column].imshow(np.clip(panel, 0.0, 1.0))
            axes[row_index, column].axis("off")
        error_reference = np.abs(reference - target).mean(axis=2)[crop]
        error_candidate = np.abs(candidate - target).mean(axis=2)[crop]
        axes[row_index, 3].imshow(error_reference, cmap="magma", vmin=0.0, vmax=vmax)
        axes[row_index, 4].imshow(error_candidate, cmap="magma", vmin=0.0, vmax=vmax)
        axes[row_index, 3].axis("off")
        axes[row_index, 4].axis("off")
        axes[row_index, 0].text(
            0.02, 0.98, f"{label} · {image_id} · Δ={delta:+.3f} dB",
            transform=axes[row_index, 0].transAxes, ha="left", va="top", fontsize=9,
            color="white", bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        )
    titles = [
        "Target (edge-rich crop)", ARMS[control]["label"], ARMS[PRIMARY_ARM]["label"],
        "Control |error|", "Tensor |error|",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=10)
    figure.suptitle(
        f"F9. Cases selected by paired PSNR-delta quantiles at {target_rate:g} bpp; "
        f"shared error scale [0, {vmax:.3f}]",
        fontsize=14, y=0.975,
    )
    _figure_scope_banner(figure, manifest)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def generate_publication_figures(
    manifest: dict[str, Any],
    data_root: Path,
    outdir: Path,
    candidate_rows: Sequence[dict[str, Any]],
    selected_rows: Sequence[dict[str, Any]],
    analysis: dict[str, Any],
    *,
    device: str,
) -> None:
    """Generate F5--F9 from measured rows; missing evidence becomes an explicit panel."""
    figure_dir = outdir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_f5(
        manifest, data_root, outdir, candidate_rows, selected_rows, analysis, device,
        figure_dir / "f5_causal_allocation.png",
    )
    _plot_f6(
        manifest, outdir, candidate_rows, selected_rows,
        figure_dir / "f6_actual_rate_phase_diagram.png",
    )
    _plot_f7(
        manifest, selected_rows, analysis, figure_dir / "f7_mechanism.png",
    )
    _plot_f8(
        manifest, outdir, candidate_rows, selected_rows, analysis,
        figure_dir / "f8_resources.png",
    )
    _plot_f9(
        manifest, data_root, outdir, selected_rows, analysis,
        figure_dir / "f9_qualitative_quantiles.png",
    )


def generate_html_index(
    manifest: dict[str, Any],
    outdir: Path,
    selected_rows: Sequence[dict[str, Any]],
    analysis: dict[str, Any],
) -> None:
    """Write a self-contained, portable audit index with only relative links."""
    cards = []
    for filename, title in FIGURE_SPECS.items():
        figure = outdir / "figures" / filename
        if figure.exists():
            cards.append(
                f'<section class="card"><h2>{html.escape(title)}</h2>'
                f'<a href="figures/{html.escape(filename)}">'
                f'<img src="figures/{html.escape(filename)}" alt="{html.escape(title)}"></a></section>'
            )
    selected_ok = [row for row in selected_rows if row.get("selection_status") == "ok"]
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['image_id']))}</td>"
        f"<td>{html.escape(ARMS[row['arm']]['label'])}</td>"
        f"<td>{float(row['target_bpp']):g}</td>"
        f"<td>{float(row['bpp']):.4f}</td>"
        f"<td>{float(row['psnr']):.3f}</td>"
        f"<td>{int(row['bytes']):,}</td>"
        f"<td><a href=\"{html.escape(row['reconstruction_path'])}\">PNG</a></td>"
        "</tr>"
        for row in selected_ok
    )
    guard = html.escape(analysis["interpretation_guard"])
    gate = html.escape(json.dumps(analysis.get("gate", {}), indent=2, sort_keys=True))
    completion = html.escape(json.dumps(analysis.get("completion", {}), indent=2, sort_keys=True))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StructSplat BENCH-007 — {html.escape(manifest['stage'])}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:0 auto;padding:24px;color:#172033;background:#f8fafc}}
h1,h2{{color:#0f172a}} .guard{{padding:14px 18px;background:#fff7ed;border-left:5px solid #ea580c;font-weight:650}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px}} .card{{background:white;padding:16px;border:1px solid #dbe3ee;border-radius:10px}}
img{{max-width:100%;height:auto}} table{{width:100%;border-collapse:collapse;background:white}} th,td{{padding:7px;border:1px solid #dbe3ee;text-align:right}} th:nth-child(-n+2),td:nth-child(-n+2){{text-align:left}}
pre{{white-space:pre-wrap;background:#0f172a;color:#e2e8f0;padding:14px;border-radius:8px}} a{{color:#1d4ed8}}
</style></head><body>
<h1>BENCH-007 actual-rate structure phase diagram</h1>
<p class="guard">{guard}</p>
<p>Stage: <strong>{html.escape(manifest['stage'])}</strong> · manifest <code>{html.escape(manifest['manifest_sha256'])}</code></p>
<p><a href="frozen_manifest.json">Frozen manifest</a> · <a href="analysis/summary.json">Analysis JSON</a> · <a href="analysis/raw_candidates.csv">Raw candidates CSV</a> · <a href="analysis/selected.csv">Selected CSV</a> · <a href="journals/candidates.jsonl">Candidate journal</a></p>
<div class="grid">{''.join(cards)}</div>
<h2>Completion</h2><pre>{completion}</pre><h2>Preregistered gate</h2><pre>{gate}</pre>
<h2>Exact-cap selected rows</h2><table><thead><tr><th>Image</th><th>Arm</th><th>Target bpp</th><th>Actual bpp</th><th>PSNR</th><th>Bytes</th><th>Recon</th></tr></thead><tbody>{table_rows}</tbody></table>
</body></html>"""
    _atomic_bytes(outdir / "index.html", document.encode("utf-8"))


def benchmark_status(manifest_path: str | Path, outdir: str | Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    output = Path(outdir)
    fits = _latest_rows(output / "journals" / "fits.jsonl", "fit_key")
    candidates = _latest_rows(output / "journals" / "candidates.jsonl", "candidate_key")
    plan = dry_run_plan(manifest)
    fit_ok = sum(
        row.get("status") == "ok" and row.get("manifest_sha256") == manifest["manifest_sha256"]
        for row in fits.values()
    )
    fit_failed = sum(
        row.get("status") == "failed" and row.get("manifest_sha256") == manifest["manifest_sha256"]
        for row in fits.values()
    )
    candidate_ok = sum(
        row.get("status") == "ok" and row.get("manifest_sha256") == manifest["manifest_sha256"]
        for row in candidates.values()
    )
    candidate_failed = sum(
        row.get("status") == "failed" and row.get("manifest_sha256") == manifest["manifest_sha256"]
        for row in candidates.values()
    )
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "stage": manifest["stage"],
        "fits": {
            "ok": fit_ok,
            "failed": fit_failed,
            "expected": plan["fits"],
            "remaining": max(0, plan["fits"] - fit_ok - fit_failed),
        },
        "candidates": {
            "ok": candidate_ok,
            "failed": candidate_failed,
            "expected": plan["codec_candidates"],
            "remaining": max(0, plan["codec_candidates"] - candidate_ok - candidate_failed),
        },
        "analysis_exists": (output / "analysis" / "summary.json").exists(),
        "index_exists": (output / "index.html").exists(),
    }


def _parse_bit_mix(value: str) -> tuple[int, int, int, int]:
    pieces = value.replace(",", "/").split("/")
    if len(pieces) != 4:
        raise argparse.ArgumentTypeError("bit mix must be means/scales/rotation/colors")
    try:
        parsed = tuple(int(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid bit mix {value!r}") from exc
    if any(piece <= 0 or piece > 24 for piece in parsed):
        raise argparse.ArgumentTypeError("bit widths must be in [1, 24]")
    return parsed  # type: ignore[return-value]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BENCH-007 actual-rate phase diagram, gate, and paper figures"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze source hashes and equal candidate grid")
    freeze.add_argument("--stage", choices=("stage0a", "stage1", "stage2"), required=True)
    freeze.add_argument("--data-root", required=True)
    freeze.add_argument("--images", nargs="+", required=True)
    freeze.add_argument("--manifest", required=True)
    freeze.add_argument("--bytes-per-gaussian", type=float, required=True)
    freeze.add_argument("--targets", type=float, nargs="+")
    freeze.add_argument("--multipliers", type=float, nargs="+", default=list(DEFAULT_MULTIPLIERS))
    freeze.add_argument(
        "--bit-mixes", type=_parse_bit_mix, nargs="+", default=list(DEFAULT_BIT_MIXES)
    )
    freeze.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    freeze.add_argument("--seed", type=int, default=0)
    freeze.add_argument("--iters", type=int, default=1500)
    freeze.add_argument("--qat-iters", type=int, default=0)
    freeze.add_argument("--renderer", choices=BENCHMARK_RENDERERS, default="normalized")
    freeze.add_argument("--render-chunk", type=int, default=512)
    freeze.add_argument("--min-gaussians", type=int, default=64)
    freeze.add_argument("--max-gaussians-per-pixel", type=float, default=0.25)
    freeze.add_argument("--no-lpips", action="store_true")
    freeze.add_argument(
        "--allow-stage0a-subset", action="store_true",
        help="plumbing tests only; forbidden for Stage 1/2",
    )

    plan = subparsers.add_parser("plan", help="print the frozen workload without executing it")
    plan.add_argument("--manifest", required=True)

    run = subparsers.add_parser("run", help="execute or resume frozen fit/codec cells")
    run.add_argument("--manifest", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--outdir", required=True)
    run.add_argument("--device")
    run.add_argument("--arms", nargs="+", choices=tuple(ARMS))
    run.add_argument("--images", nargs="+")
    run.add_argument("--max-new-fits", type=int)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument(
        "--revalidate-candidates",
        action="store_true",
        help="re-encode and revalidate every candidate without refitting fields",
    )

    analyze = subparsers.add_parser("analyze", help="RDO-select, test gate, and generate F5--F9")
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--data-root", required=True)
    analyze.add_argument("--outdir", required=True)
    analyze.add_argument("--device")
    analyze.add_argument("--no-figures", action="store_true")

    conventional = subparsers.add_parser(
        "conventional", help="sweep context-only PNG/JPEG/AVIF actual streams"
    )
    conventional.add_argument("--manifest", required=True)
    conventional.add_argument("--data-root", required=True)
    conventional.add_argument("--outdir", required=True)
    conventional.add_argument("--jpeg-qualities", type=int, nargs="+")
    conventional.add_argument("--avif-qualities", type=int, nargs="+")
    conventional.add_argument("--allow-no-modern-codec", action="store_true")

    calibrate = subparsers.add_parser(
        "calibrate", help="run Stage-0b bytes/G calibration on the four frozen DIV2K IDs"
    )
    calibrate.add_argument("--data-root", required=True)
    calibrate.add_argument("--images", nargs="+", required=True)
    calibrate.add_argument("--outdir", required=True)
    calibrate.add_argument(
        "--count-densities-per-megapixel", type=float, nargs="+", default=[2048.0, 8192.0]
    )
    calibrate.add_argument("--iters", type=int, default=50)
    calibrate.add_argument("--bit-mix", type=_parse_bit_mix, default=(16, 8, 8, 8))
    calibrate.add_argument("--device")
    calibrate.add_argument("--renderer", choices=BENCHMARK_RENDERERS, default="normalized")
    calibrate.add_argument("--render-chunk", type=int, default=512)

    status = subparsers.add_parser("status", help="inspect durable journal progress")
    status.add_argument("--manifest", required=True)
    status.add_argument("--outdir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        payload = freeze_manifest(
            args.images,
            args.data_root,
            args.manifest,
            stage=args.stage,
            bytes_per_gaussian=args.bytes_per_gaussian,
            targets=args.targets,
            multipliers=args.multipliers,
            bit_mixes=args.bit_mixes,
            arms=args.arms,
            seed=args.seed,
            iters=args.iters,
            qat_iters=args.qat_iters,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
            min_gaussians=args.min_gaussians,
            max_gaussians_per_pixel=args.max_gaussians_per_pixel,
            compute_lpips=not args.no_lpips,
            enforce_stage_ids=not args.allow_stage0a_subset,
        )
        print(json.dumps({
            "manifest": str(Path(args.manifest)),
            "manifest_sha256": payload["manifest_sha256"],
            "plan": dry_run_plan(payload),
        }, indent=2))
    elif args.command == "plan":
        print(json.dumps(dry_run_plan(load_manifest(args.manifest)), indent=2))
    elif args.command == "run":
        print(json.dumps(run_manifest(
            args.manifest,
            args.data_root,
            args.outdir,
            device=args.device,
            only_arms=args.arms,
            only_images=args.images,
            max_new_fits=args.max_new_fits,
            retry_failed=args.retry_failed,
            revalidate_candidates=args.revalidate_candidates,
        ), indent=2))
    elif args.command == "analyze":
        print(json.dumps(analyze_manifest(
            args.manifest,
            args.data_root,
            args.outdir,
            device=args.device,
            make_figures=not args.no_figures,
        ), indent=2))
    elif args.command == "conventional":
        kwargs: dict[str, Any] = {}
        if args.jpeg_qualities:
            kwargs["jpeg_qualities"] = args.jpeg_qualities
        if args.avif_qualities:
            kwargs["avif_qualities"] = args.avif_qualities
        result = sweep_conventional_codecs(
            args.manifest,
            args.data_root,
            args.outdir,
            require_modern=not args.allow_no_modern_codec,
            **kwargs,
        )
        print(json.dumps({
            "rows": len(result["rows"]),
            "modern_codec_available": result["modern_codec_available"],
        }, indent=2))
    elif args.command == "calibrate":
        result = calibrate_bytes_per_gaussian(
            args.images,
            args.data_root,
            args.outdir,
            count_densities_per_megapixel=args.count_densities_per_megapixel,
            iters=args.iters,
            bit_mix=args.bit_mix,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        print(json.dumps({
            "recommended_bytes_per_gaussian": result["recommended_bytes_per_gaussian"],
            "range_bytes_per_gaussian": result["range_bytes_per_gaussian"],
        }, indent=2))
    elif args.command == "status":
        print(json.dumps(benchmark_status(args.manifest, args.outdir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
