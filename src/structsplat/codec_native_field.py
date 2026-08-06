"""CORE-016 codec-native dual-plane Gaussian observation field (ADR-0032).

This default-off reference representation separates two jobs that an explicit Gaussian row often
mixes together:

* appearance is a conventional encoded RGB raster interpreted as an *implicit normalized Gaussian
  lattice* at query time; and
* structure is a small explicit anisotropic Gaussian measure with independent non-negative mass.

The conventional image payload is part of the packet and every byte is charged.  It is not a free
source-image side channel.  The module is NumPy/Pillow-only and does not import torch or realtime-gs;
the optional consumer adapter lives in :mod:`structsplat.realtime_gs_adapter`.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Literal
import zipfile

import numpy as np
from PIL import Image

from .config import StructureTensorConfig
from .density import density_from_energy, warp_unit_points
from .observation_field import (
    AlphaSemantics,
    CanvasCropTransform,
    FieldSemantics,
    ObservationField2D,
    SupportSemantics,
    load_observation_field,
    pack_alpha,
)
from .sampling import halton_unit
from .structure_tensor import compute as compute_structure_tensor


PACKET_SCHEMA = "structsplat.codec_native_dual_plane.v2"
PACKET_EXTENSION = ".sgdp"
_MANIFEST_NAME = "manifest.json"
_STRUCTURE_NAME = "structure.of2d"
_APPEARANCE_NAME = "appearance.bin"
_MEMBER_NAMES = frozenset({_MANIFEST_NAME, _STRUCTURE_NAME, _APPEARANCE_NAME})
_MAX_PACKET_BYTES = 1 << 30
_MAX_MEMBER_BYTES = 1 << 30
_MAX_PIXELS = 100_000_000
_MAX_STRUCTURE_ROWS = 2_000_000

AppearanceCodec = Literal["jpeg", "webp", "webp_lossless"]
CoordinateSpace = Literal["crop", "canvas"]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(payload: bytes) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} is forbidden")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("packet manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("packet manifest must be a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        relation = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {relation}")
    return result


def _strict_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


@dataclass(frozen=True)
class CodecNativeFieldConfig:
    """Configuration embedded in every packet.

    ``lattice_sigma_px`` is intentionally small: the decoded raster supplies the coefficients and
    the Gaussian lattice supplies a continuous extension without materially blurring pixel-center
    replay.  A radius of two already drops less than machine-relevant mass at the default sigma.
    """

    appearance_codec: AppearanceCodec = "webp"
    appearance_quality: int = 90
    lattice_sigma_px: float = 0.25
    lattice_radius_px: int = 2
    lattice_prefilter_steps: int = 0
    structural_count: int = 2048
    structural_seed: int = 0
    structural_density_base: float = 0.08
    structural_density_power: float = 0.7
    structural_scale_factor: float = 0.65
    structural_max_aspect: float = 4.0

    def __post_init__(self) -> None:
        if self.appearance_codec not in {"jpeg", "webp", "webp_lossless"}:
            raise ValueError(
                "appearance_codec must be 'jpeg', 'webp', or 'webp_lossless'"
            )
        quality = _strict_int(self.appearance_quality, "appearance_quality", minimum=1)
        if quality > 100:
            raise ValueError("appearance_quality must be <= 100")
        object.__setattr__(self, "appearance_quality", quality)
        sigma = _finite_float(self.lattice_sigma_px, "lattice_sigma_px", positive=True)
        object.__setattr__(self, "lattice_sigma_px", sigma)
        radius = _strict_int(self.lattice_radius_px, "lattice_radius_px", minimum=1)
        if radius > 16:
            raise ValueError("lattice_radius_px must be <= 16")
        object.__setattr__(self, "lattice_radius_px", radius)
        prefilter_steps = _strict_int(
            self.lattice_prefilter_steps, "lattice_prefilter_steps", minimum=0
        )
        if prefilter_steps > 64:
            raise ValueError("lattice_prefilter_steps must be <= 64")
        if prefilter_steps:
            off_diagonal_sum = 2.0 * sum(
                math.exp(-0.5 * (offset / sigma) ** 2)
                for offset in range(1, radius + 1)
            )
            if off_diagonal_sum >= 1.0:
                raise ValueError(
                    "prefiltered Gaussian lattice requires a strictly diagonal-dominant "
                    f"kernel; off-diagonal sum is {off_diagonal_sum:.6f}"
                )
        object.__setattr__(self, "lattice_prefilter_steps", prefilter_steps)
        object.__setattr__(
            self,
            "structural_count",
            _strict_int(self.structural_count, "structural_count", minimum=1),
        )
        object.__setattr__(
            self,
            "structural_seed",
            _strict_int(self.structural_seed, "structural_seed", minimum=0),
        )
        for name in (
            "structural_density_base",
            "structural_density_power",
            "structural_scale_factor",
            "structural_max_aspect",
        ):
            value = _finite_float(getattr(self, name), name, positive=True)
            object.__setattr__(self, name, value)
        if self.structural_density_base > 1.0:
            raise ValueError("structural_density_base must be <= 1")
        if self.structural_max_aspect < 1.0:
            raise ValueError("structural_max_aspect must be >= 1")

    @classmethod
    def from_record(cls, value: object) -> "CodecNativeFieldConfig":
        if not isinstance(value, Mapping):
            raise TypeError("packet config must be an object")
        expected = set(cls.__dataclass_fields__)
        _require_exact_keys(value, expected, "packet config")
        return cls(**{name: value[name] for name in expected})


@dataclass(frozen=True)
class CodecNativeQuery:
    """One dual-plane point query."""

    color: np.ndarray
    structural_density: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class PacketByteLedger:
    """Exact accounting for a serialized packet.

    Member values are their physical compressed sizes in the outer container.  The difference to
    ``complete_bytes`` is ZIP framing; raw member sizes remain independently bound in the manifest.
    """

    complete_bytes: int
    manifest_bytes: int
    appearance_bytes: int
    structure_bytes: int
    container_overhead_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.complete_bytes,
            self.manifest_bytes,
            self.appearance_bytes,
            self.structure_bytes,
            self.container_overhead_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("byte-ledger values must be non-negative")
        if self.complete_bytes != sum(values[1:]):
            raise ValueError("packet byte ledger does not sum to complete_bytes")


def _as_rgb_float(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError("image must have shape (H,W,3+)")
    value = value[..., :3]
    if value.dtype == np.uint8:
        result = value.astype(np.float32) / 255.0
    elif value.dtype.kind == "f":
        result = np.asarray(value, dtype=np.float32)
    else:
        raise TypeError("image must use uint8 or floating RGB values")
    if not np.isfinite(result).all() or ((result < 0.0) | (result > 1.0)).any():
        raise ValueError("floating image values must be finite and in [0,1]")
    return np.ascontiguousarray(result)


def _as_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    value = np.asarray(mask)
    if value.shape != shape:
        raise ValueError(f"mask must have shape {shape}")
    if value.dtype == np.bool_:
        return np.ascontiguousarray(value)
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError("mask must be finite bool or numeric data")
    if ((value < 0) | (value > 1)).any():
        if value.dtype == np.uint8:
            value = value.astype(np.float32) / 255.0
        else:
            raise ValueError("numeric mask must lie in [0,1]")
    return np.ascontiguousarray(value >= 0.5)


def encode_appearance(image: np.ndarray, config: CodecNativeFieldConfig) -> bytes:
    """Encode one RGB coefficient raster with a charged conventional codec."""
    rgb = _as_rgb_float(image)
    pixels = np.rint(rgb * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(pixels, mode="RGB")
    stream = io.BytesIO()
    if config.appearance_codec == "jpeg":
        pil.save(
            stream,
            format="JPEG",
            quality=config.appearance_quality,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
    elif config.appearance_codec == "webp":
        pil.save(
            stream,
            format="WEBP",
            quality=config.appearance_quality,
            method=4,
            exact=True,
        )
    else:
        # In lossless WebP mode Pillow interprets quality as compression effort.  It does not
        # alter decoded RGB, so the same packet grammar spans a fast/high-ratio effort ladder.
        pil.save(
            stream,
            format="WEBP",
            lossless=True,
            quality=config.appearance_quality,
            method=6,
            exact=True,
        )
    return stream.getvalue()


def decode_appearance(
    payload: bytes,
    codec: AppearanceCodec,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Strictly decode the charged appearance payload to float32 RGB."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("appearance payload must be non-empty bytes")
    if len(payload) > _MAX_MEMBER_BYTES:
        raise ValueError("appearance payload exceeds the decoder byte cap")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            expected_format = "JPEG" if codec == "jpeg" else "WEBP"
            if image.format != expected_format:
                raise ValueError(
                    f"appearance payload format {image.format!r} does not match {codec!r}"
                )
            image.load()
            if image.size != (expected_shape[1], expected_shape[0]):
                raise ValueError("appearance payload dimensions do not match packet metadata")
            result = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except (OSError, SyntaxError) as exc:
        raise ValueError("cannot decode appearance payload") from exc
    return np.ascontiguousarray(result)


def _gaussian_lattice_weights(config: CodecNativeFieldConfig) -> np.ndarray:
    offsets = np.arange(1, config.lattice_radius_px + 1, dtype=np.float64)
    return np.exp(-0.5 * np.square(offsets / config.lattice_sigma_px)).astype(np.float32)


def _axis_normalizer(length: int, weights: np.ndarray) -> np.ndarray:
    result = np.ones(length, dtype=np.float32)
    for offset, weight in enumerate(weights, start=1):
        if offset >= length:
            break
        result[offset:] += weight
        result[:-offset] += weight
    return result


def _jacobi_axis_solve(
    right_hand_side: np.ndarray,
    weights: np.ndarray,
    *,
    axis: int,
    steps: int,
) -> np.ndarray:
    """Solve one strictly diagonal-dominant finite Gaussian convolution system.

    The diagonal is one.  At the supported prefilter widths, the off-diagonal row sum is below
    one, so plain Jacobi is deterministic and rapidly convergent.  Keeping this small solver in
    NumPy avoids adding SciPy to the cold packet decoder.
    """
    current = np.ascontiguousarray(right_hand_side, dtype=np.float32).copy()
    next_value = np.empty_like(current)
    off_diagonal = np.empty_like(current)
    for _ in range(steps):
        off_diagonal.fill(0.0)
        for offset, weight in enumerate(weights, start=1):
            if offset >= current.shape[axis]:
                break
            left = [slice(None)] * current.ndim
            right = [slice(None)] * current.ndim
            left[axis] = slice(offset, None)
            right[axis] = slice(None, -offset)
            off_diagonal[tuple(left)] += weight * current[tuple(right)]
            off_diagonal[tuple(right)] += weight * current[tuple(left)]
        np.subtract(right_hand_side, off_diagonal, out=next_value)
        current, next_value = next_value, current
    return np.ascontiguousarray(current)


def prefilter_appearance_coefficients(
    decoded: np.ndarray,
    config: CodecNativeFieldConfig,
) -> np.ndarray:
    """Derive signed fixed-Gaussian coefficients that interpolate decoded pixel centers.

    For the separable normalized lattice, integer-center interpolation requires

    ``K_y C K_x^T = Y * (K_y 1) * (K_x 1)^T``.

    Two bounded Jacobi solves recover ``C``.  The payload remains the conventional encoded
    raster ``Y``; these coefficients are deterministic decoder state and may leave ``[0,1]``.
    """
    samples = _as_rgb_float(decoded)
    if config.lattice_prefilter_steps == 0:
        return samples
    weights = _gaussian_lattice_weights(config)
    off_diagonal_sum = 2.0 * float(weights.sum())
    if off_diagonal_sum >= 1.0:
        raise ValueError(
            "prefiltered Gaussian lattice requires a strictly diagonal-dominant kernel; "
            f"off-diagonal sum is {off_diagonal_sum:.6f}"
        )
    height, width = samples.shape[:2]
    normalizer_y = _axis_normalizer(height, weights)
    normalizer_x = _axis_normalizer(width, weights)
    right_hand_side = (
        samples
        * normalizer_y[:, None, None]
        * normalizer_x[None, :, None]
    ).astype(np.float32)
    solved_y = _jacobi_axis_solve(
        right_hand_side,
        weights,
        axis=0,
        steps=config.lattice_prefilter_steps,
    )
    return _jacobi_axis_solve(
        solved_y,
        weights,
        axis=1,
        steps=config.lattice_prefilter_steps,
    )


def build_structural_field(
    image: np.ndarray,
    *,
    config: CodecNativeFieldConfig,
    canvas_crop: CanvasCropTransform | None = None,
    mask: np.ndarray | None = None,
    structure_config: StructureTensorConfig | None = None,
) -> ObservationField2D:
    """Build an exact-count, seed-reproducible structural Gaussian measure.

    Positions come from a Cranley--Patterson-shifted Halton set warped through the structure
    density.  This is O(pixels + rows*width/chunk), avoids WSE's quadratic candidate competition,
    and deliberately does not claim blue-noise optimality.  Appearance coefficients are zero: the
    encoded lattice, not this measure, is authoritative for color.
    """
    rgb = _as_rgb_float(image)
    height, width = rgb.shape[:2]
    if height * width > _MAX_PIXELS:
        raise ValueError("image exceeds the structural decoder pixel cap")
    mask_value = _as_mask(mask, (height, width))
    if mask_value is not None and not bool(mask_value.any()):
        raise ValueError("structural mask must contain at least one pixel")
    if canvas_crop is None:
        canvas_crop = CanvasCropTransform(width, height, 0, 0, width, height)
    if (canvas_crop.crop_height, canvas_crop.crop_width) != (height, width):
        raise ValueError("canvas_crop dimensions must match the supplied crop image")

    tensor = compute_structure_tensor(rgb, structure_config)
    density = density_from_energy(
        tensor.energy,
        config.structural_density_base,
        config.structural_density_power,
        tensor.energy_ref,
    )
    if mask_value is not None:
        density = density * mask_value.astype(np.float64)
        total = float(density.sum())
        if total <= 0.0:
            raise ValueError("masked structural density is empty")
        density = density / total

    rng = np.random.default_rng(config.structural_seed)
    positions = warp_unit_points(halton_unit(config.structural_count, rng), density)
    ix = np.rint(positions[:, 0]).astype(np.int64).clip(0, width - 1)
    iy = np.rint(positions[:, 1]).astype(np.int64).clip(0, height - 1)
    if mask_value is not None:
        outside = ~mask_value[iy, ix]
        if bool(outside.any()):
            # Vanishing-density roundoff can only affect the nearest pixel at a boundary.  Use a
            # deterministic exact-density rank fallback rather than silently retaining an outside
            # proposal.
            ranked = np.argsort(density.reshape(-1), kind="stable")[::-1]
            replacements = ranked[: int(outside.sum())]
            replacement_y, replacement_x = np.divmod(replacements, width)
            positions[outside, 0] = replacement_x
            positions[outside, 1] = replacement_y
            ix[outside] = replacement_x
            iy[outside] = replacement_y

    active_area = int(mask_value.sum()) if mask_value is not None else height * width
    spacing = math.sqrt(max(active_area, 1) / config.structural_count)
    coherence = np.sqrt(np.clip(tensor.coherence[iy, ix], 0.0, 1.0)).astype(np.float64)
    aspect = 1.0 + (config.structural_max_aspect - 1.0) * coherence
    base_scale = max(0.35, config.structural_scale_factor * spacing)
    along = np.maximum(0.35, base_scale * np.sqrt(aspect))
    across = np.maximum(0.35, base_scale / np.sqrt(aspect))
    scales = np.stack([along, across], axis=1).astype(np.float32)
    rotations = tensor.along_edge_angle[iy, ix].astype(np.float32)

    reference = max(float(tensor.energy_ref or 0.0), 1e-12)
    normalized_energy = np.clip(tensor.energy[iy, ix].astype(np.float64) / reference, 0.0, 1.0)
    mass = (0.1 + np.sqrt(normalized_energy)).astype(np.float32)
    mass /= max(float(mass.mean()), 1e-12)

    alpha_semantics = AlphaSemantics()
    packed_alpha = None
    if mask_value is not None:
        packed_alpha = pack_alpha(mask_value)
        alpha_semantics = AlphaSemantics(
            payload_encoding="binary_exact_packbits_little",
            matting_mode="multiply_alpha",
            boundary_policy="unconstrained",
        )
    semantics = FieldSemantics(
        coefficient_domain="signed",
        support=SupportSemantics(mode="axis_aligned_bbox", sigma_cutoff=3.0),
        alpha=alpha_semantics,
    )
    return ObservationField2D(
        means_xy=positions.astype(np.float32),
        log_scales_xy=np.log(scales),
        rotations_rad=rotations,
        rgb_coeff=np.zeros((config.structural_count, 3), dtype=np.float32),
        structural_mass=mass,
        canvas_crop=canvas_crop,
        semantics=semantics,
        packed_alpha=packed_alpha,
    )


def _serialize_structure(field_value: ObservationField2D) -> bytes:
    with tempfile.TemporaryDirectory(prefix="structsplat-sgdp-structure-") as directory:
        path = Path(directory) / _STRUCTURE_NAME
        field_value.save_lossless(path)
        return path.read_bytes()


def _deserialize_structure(payload: bytes) -> ObservationField2D:
    if len(payload) > _MAX_MEMBER_BYTES:
        raise ValueError("structure payload exceeds the decoder byte cap")
    with tempfile.TemporaryDirectory(prefix="structsplat-sgdp-structure-") as directory:
        path = Path(directory) / _STRUCTURE_NAME
        path.write_bytes(payload)
        result = load_observation_field(path)
    if result.n > _MAX_STRUCTURE_ROWS:
        raise ValueError("structure payload exceeds the decoder row cap")
    return result


@dataclass(frozen=True)
class CodecNativeField:
    """Cold-decodable dual-plane observation packet in memory."""

    appearance_payload: bytes
    structure: ObservationField2D
    config: CodecNativeFieldConfig
    source_sha256: str
    source_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.appearance_payload, bytes) or not self.appearance_payload:
            raise ValueError("appearance_payload must be non-empty bytes")
        if not isinstance(self.structure, ObservationField2D):
            raise TypeError("structure must be an ObservationField2D")
        if self.structure.structural_mass is None:
            raise ValueError("structure field must carry independent structural_mass")
        if self.structure.n != self.config.structural_count:
            raise ValueError("structure row count does not match packet config")
        if not isinstance(self.source_sha256, str) or (
            len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "source_bytes", _strict_int(self.source_bytes, "source_bytes"))
        # Decode once at construction so a corrupt or mismatched image cannot survive until the
        # first query.
        _ = self.decoded_appearance

    @property
    def canvas_crop(self) -> CanvasCropTransform:
        return self.structure.canvas_crop

    @property
    def crop_shape(self) -> tuple[int, int]:
        return self.structure.crop_shape

    @cached_property
    def decoded_appearance(self) -> np.ndarray:
        return decode_appearance(
            self.appearance_payload,
            self.config.appearance_codec,
            self.crop_shape,
        )

    @cached_property
    def appearance_coefficients(self) -> np.ndarray:
        return prefilter_appearance_coefficients(self.decoded_appearance, self.config)

    @cached_property
    def alpha_mask(self) -> np.ndarray:
        if self.structure.packed_alpha is None:
            return np.ones(self.crop_shape, dtype=bool)
        return np.ascontiguousarray(self.structure.alpha_mask())

    def _crop_points(self, points_xy: object, coordinate_space: CoordinateSpace) -> np.ndarray:
        points = np.asarray(points_xy)
        if points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in "fiu":
            raise ValueError("points_xy must have numeric shape (M,2)")
        points = np.asarray(points, dtype=np.float64)
        if not np.isfinite(points).all():
            raise ValueError("points_xy must be finite")
        if coordinate_space == "crop":
            return points
        if coordinate_space == "canvas":
            return self.canvas_crop.canvas_to_crop(points)
        raise ValueError("coordinate_space must be 'crop' or 'canvas'")

    def query_appearance(
        self,
        points_xy: object,
        *,
        coordinate_space: CoordinateSpace = "crop",
        apply_alpha: bool = True,
        chunk_size: int = 262_144,
    ) -> np.ndarray:
        """Evaluate the normalized implicit Gaussian lattice at arbitrary coordinates."""
        points = self._crop_points(points_xy, coordinate_space)
        chunk = _strict_int(chunk_size, "chunk_size", minimum=1)
        height, width = self.crop_shape
        sigma2 = self.config.lattice_sigma_px**2
        radius = self.config.lattice_radius_px
        offsets_y, offsets_x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        offsets_x = offsets_x.reshape(1, -1)
        offsets_y = offsets_y.reshape(1, -1)
        result = np.zeros((points.shape[0], 3), dtype=np.float32)
        for start in range(0, points.shape[0], chunk):
            end = min(start + chunk, points.shape[0])
            local = points[start:end]
            base_x = np.floor(local[:, 0]).astype(np.int64)[:, None]
            base_y = np.floor(local[:, 1]).astype(np.int64)[:, None]
            raw_x = base_x + offsets_x
            raw_y = base_y + offsets_y
            present = (raw_x >= 0) & (raw_x < width) & (raw_y >= 0) & (raw_y < height)
            gather_x = raw_x.clip(0, width - 1)
            gather_y = raw_y.clip(0, height - 1)
            dx = local[:, None, 0] - raw_x
            dy = local[:, None, 1] - raw_y
            weights = np.exp(-0.5 * (dx * dx + dy * dy) / sigma2) * present
            denominator = weights.sum(axis=1)
            valid = (
                (local[:, 0] >= -0.5)
                & (local[:, 0] < width - 0.5)
                & (local[:, 1] >= -0.5)
                & (local[:, 1] < height - 0.5)
                & (denominator > np.finfo(np.float64).tiny)
            )
            values = self.appearance_coefficients[gather_y, gather_x]
            color = (weights[..., None] * values).sum(axis=1)
            color /= np.maximum(denominator[:, None], np.finfo(np.float64).tiny)
            color[~valid] = 0.0
            if apply_alpha:
                nearest_x = np.floor(local[:, 0] + 0.5).astype(np.int64).clip(0, width - 1)
                nearest_y = np.floor(local[:, 1] + 0.5).astype(np.int64).clip(0, height - 1)
                alpha = self.alpha_mask[nearest_y, nearest_x] & valid
                color *= alpha[:, None]
            result[start:end] = color.astype(np.float32)
        return result

    def query(
        self,
        points_xy: object,
        *,
        coordinate_space: CoordinateSpace = "crop",
        chunk_size: int = 262_144,
    ) -> CodecNativeQuery:
        points = self._crop_points(points_xy, coordinate_space)
        height, width = self.crop_shape
        valid = (
            (points[:, 0] >= -0.5)
            & (points[:, 0] < width - 0.5)
            & (points[:, 1] >= -0.5)
            & (points[:, 1] < height - 0.5)
        )
        nearest_x = np.floor(points[:, 0] + 0.5).astype(np.int64).clip(0, width - 1)
        nearest_y = np.floor(points[:, 1] + 0.5).astype(np.int64).clip(0, height - 1)
        alpha = self.alpha_mask[nearest_y, nearest_x] & valid
        color = self.query_appearance(
            points,
            coordinate_space="crop",
            apply_alpha=True,
            chunk_size=chunk_size,
        )
        density = self.structure.structural_density(
            points,
            coordinate_space="crop",
            apply_alpha=self.structure.packed_alpha is not None,
        )
        density[~valid] = 0.0
        return CodecNativeQuery(
            color=color,
            structural_density=density.astype(np.float32),
            alpha=alpha,
            valid=valid,
        )

    def render(self, *, apply_alpha: bool = True, row_chunk: int = 64) -> np.ndarray:
        height, width = self.crop_shape
        chunk_rows = _strict_int(row_chunk, "row_chunk", minimum=1)
        result = np.empty((height, width, 3), dtype=np.float32)
        x = np.arange(width, dtype=np.float64)
        for y_start in range(0, height, chunk_rows):
            y_end = min(y_start + chunk_rows, height)
            yy, xx = np.meshgrid(
                np.arange(y_start, y_end, dtype=np.float64), x, indexing="ij"
            )
            points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
            values = self.query_appearance(
                points,
                apply_alpha=apply_alpha,
                chunk_size=max(width, width * chunk_rows),
            )
            result[y_start:y_end] = values.reshape(y_end - y_start, width, 3)
        return result

    def _manifest(self, structure_payload: bytes) -> dict[str, object]:
        crop = self.canvas_crop.to_record()
        members = {
            _APPEARANCE_NAME: {
                "bytes": len(self.appearance_payload),
                "sha256": _sha256(self.appearance_payload),
            },
            _STRUCTURE_NAME: {
                "bytes": len(structure_payload),
                "sha256": _sha256(structure_payload),
            },
        }
        return {
            "schema": PACKET_SCHEMA,
            "config": asdict(self.config),
            "canvas_crop": crop,
            "source": {"bytes": self.source_bytes, "sha256": self.source_sha256},
            "decoded_rgb_sha256": _sha256(
                np.ascontiguousarray(
                    np.rint(self.decoded_appearance * 255.0).clip(0, 255).astype(np.uint8)
                ).tobytes()
            ),
            "appearance_coefficients_sha256": _sha256(
                np.ascontiguousarray(self.appearance_coefficients, dtype="<f4").tobytes()
            ),
            "structure_content_sha256": self.structure.canonical_hash(),
            "members": members,
        }

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> PacketByteLedger:
        target = Path(path)
        if not target.parent.is_dir():
            raise FileNotFoundError(f"packet parent directory does not exist: {target.parent}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite packet: {target}")
        structure_payload = _serialize_structure(self.structure)
        manifest_payload = _canonical_json(self._manifest(structure_payload))
        members = {
            _MANIFEST_NAME: manifest_payload,
            _APPEARANCE_NAME: self.appearance_payload,
            _STRUCTURE_NAME: structure_payload,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(temporary, "w") as archive:
                for name, payload in sorted(members.items()):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    compression = (
                        zipfile.ZIP_STORED if name == _APPEARANCE_NAME else zipfile.ZIP_DEFLATED
                    )
                    info.compress_type = compression
                    info.external_attr = 0o644 << 16
                    archive.writestr(info, payload, compress_type=compression, compresslevel=9)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            if target.exists() and not overwrite:
                raise FileExistsError(f"refusing to overwrite packet: {target}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return packet_byte_ledger(target)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "CodecNativeField":
        source = Path(path)
        if source.stat().st_size > _MAX_PACKET_BYTES:
            raise ValueError("packet exceeds the decoder byte cap")
        try:
            with zipfile.ZipFile(source, "r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(set(names)):
                    raise ValueError("packet contains duplicate members")
                if set(names) != _MEMBER_NAMES:
                    raise ValueError("packet members do not match the v2 grammar")
                expected_compression = {
                    _APPEARANCE_NAME: zipfile.ZIP_STORED,
                    _MANIFEST_NAME: zipfile.ZIP_DEFLATED,
                    _STRUCTURE_NAME: zipfile.ZIP_DEFLATED,
                }
                if any(
                    info.flag_bits & 0x1
                    or info.compress_type != expected_compression[info.filename]
                    or info.file_size > _MAX_MEMBER_BYTES
                    or info.compress_size > _MAX_MEMBER_BYTES
                    for info in infos
                ):
                    raise ValueError("packet has encrypted, mis-coded, or oversized members")
                if sum(info.file_size for info in infos) > _MAX_PACKET_BYTES:
                    raise ValueError("packet members exceed the aggregate decoder byte cap")
                payloads = {name: archive.read(name) for name in names}
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            raise ValueError("invalid codec-native field packet") from exc

        manifest = _strict_json_object(payloads[_MANIFEST_NAME])
        _require_exact_keys(
            manifest,
            {
                "schema",
                "config",
                "canvas_crop",
                "source",
                "decoded_rgb_sha256",
                "appearance_coefficients_sha256",
                "structure_content_sha256",
                "members",
            },
            "packet manifest",
        )
        if manifest["schema"] != PACKET_SCHEMA:
            raise ValueError(f"unsupported packet schema {manifest['schema']!r}")
        config = CodecNativeFieldConfig.from_record(manifest["config"])
        canvas_crop = CanvasCropTransform.from_record(manifest["canvas_crop"])
        source_record = manifest["source"]
        if not isinstance(source_record, Mapping):
            raise TypeError("packet source record must be an object")
        _require_exact_keys(source_record, {"bytes", "sha256"}, "packet source")
        members = manifest["members"]
        if not isinstance(members, Mapping) or set(members) != {
            _APPEARANCE_NAME,
            _STRUCTURE_NAME,
        }:
            raise ValueError("packet member descriptors are incomplete")
        for name in (_APPEARANCE_NAME, _STRUCTURE_NAME):
            descriptor = members[name]
            if not isinstance(descriptor, Mapping):
                raise TypeError(f"packet descriptor {name!r} must be an object")
            _require_exact_keys(descriptor, {"bytes", "sha256"}, f"packet descriptor {name}")
            payload = payloads[name]
            if descriptor["bytes"] != len(payload) or descriptor["sha256"] != _sha256(payload):
                raise ValueError(f"packet member integrity mismatch: {name}")

        structure = _deserialize_structure(payloads[_STRUCTURE_NAME])
        if structure.canvas_crop != canvas_crop:
            raise ValueError("packet structure transform does not match the manifest")
        if structure.canonical_hash() != manifest["structure_content_sha256"]:
            raise ValueError("packet structure content digest mismatch")
        decoded = decode_appearance(
            payloads[_APPEARANCE_NAME], config.appearance_codec, structure.crop_shape
        )
        decoded_digest = _sha256(
            np.ascontiguousarray(np.rint(decoded * 255.0).clip(0, 255).astype(np.uint8)).tobytes()
        )
        if decoded_digest != manifest["decoded_rgb_sha256"]:
            raise ValueError("packet decoded RGB digest mismatch")
        result = cls(
            appearance_payload=payloads[_APPEARANCE_NAME],
            structure=structure,
            config=config,
            source_sha256=str(source_record["sha256"]),
            source_bytes=_strict_int(source_record["bytes"], "source.bytes"),
        )
        coefficient_digest = _sha256(
            np.ascontiguousarray(result.appearance_coefficients, dtype="<f4").tobytes()
        )
        if coefficient_digest != manifest["appearance_coefficients_sha256"]:
            raise ValueError("packet derived appearance-coefficient digest mismatch")
        return result


def build_codec_native_field(
    image: np.ndarray,
    *,
    config: CodecNativeFieldConfig | None = None,
    mask: np.ndarray | None = None,
    canvas_crop: CanvasCropTransform | None = None,
    source_payload: bytes | None = None,
    structure_config: StructureTensorConfig | None = None,
) -> CodecNativeField:
    """Build a packet from decoded pixels and optional exact source-file provenance."""
    cfg = config or CodecNativeFieldConfig()
    rgb = _as_rgb_float(image)
    structure = build_structural_field(
        rgb,
        config=cfg,
        canvas_crop=canvas_crop,
        mask=mask,
        structure_config=structure_config,
    )
    if source_payload is None:
        canonical = np.ascontiguousarray(np.rint(rgb * 255.0).clip(0, 255).astype(np.uint8))
        source_payload = canonical.tobytes()
    if not isinstance(source_payload, bytes):
        raise TypeError("source_payload must be bytes when supplied")
    return CodecNativeField(
        appearance_payload=encode_appearance(rgb, cfg),
        structure=structure,
        config=cfg,
        source_sha256=_sha256(source_payload),
        source_bytes=len(source_payload),
    )


def packet_byte_ledger(path: str | os.PathLike[str]) -> PacketByteLedger:
    source = Path(path)
    complete = source.stat().st_size
    try:
        with zipfile.ZipFile(source, "r") as archive:
            sizes = {info.filename: info.compress_size for info in archive.infolist()}
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid packet for byte accounting") from exc
    if set(sizes) != _MEMBER_NAMES:
        raise ValueError("packet members do not match the v2 byte ledger")
    member_total = sum(sizes.values())
    return PacketByteLedger(
        complete_bytes=complete,
        manifest_bytes=sizes[_MANIFEST_NAME],
        appearance_bytes=sizes[_APPEARANCE_NAME],
        structure_bytes=sizes[_STRUCTURE_NAME],
        container_overhead_bytes=complete - member_total,
    )
