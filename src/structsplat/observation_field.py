"""Versioned, lossless Observation Field V2 reference contract (CORE-013).

This module is intentionally NumPy-only.  It defines the semantic interchange boundary and a
small CPU oracle; production fitting and rasterization remain in the torch modules.  In
particular, importing this module does not select additive rendering, change ``GaussianField``,
or alter any CLI/default behavior.

The authoritative appearance coefficient is ``rgb_coeff``:

    F(x) = background_rgb + sum_i kernel_i(x) * rgb_coeff_i

Optional ``structural_mass`` is independent:

    S(x) = sum_i kernel_i(x) * structural_mass_i

No color/mass quotient, display clamp, alpha matte, or normalized compositor is implicit.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Literal
import zipfile

import numpy as np


SCHEMA_VERSION = "2.0.0"
REFERENCE_CONTAINER = "structsplat.observation_field.reference_npz"
REFERENCE_CONTAINER_VERSION = 1

_HASH_DOMAIN = b"StructSplat/ObservationField2D/canonical/v1\0"
_ARRAY_HASH_DOMAIN = b"StructSplat/ObservationField2D/array/v1\0"
_BASE_ARRAY_NAMES = frozenset({"means_xy", "log_scales_xy", "rotations_rad", "rgb_coeff"})
_OPTIONAL_ARRAY_NAMES = frozenset(
    {"structural_mass", "filter_variance_px2", "background_rgb", "packed_alpha"}
)
_ALL_ARRAY_NAMES = _BASE_ARRAY_NAMES | _OPTIONAL_ARRAY_NAMES

CoordinateSpace = Literal["crop", "canvas"]


def _strict_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _finite_float(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _expect_choice(value: str, name: str, choices: frozenset[str]) -> str:
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {expected}; got {value!r}")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _freeze_json(value: object, path: str = "camera.payload") -> object:
    """Validate JSON data and return an immutable, key-sorted representation."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} keys must be non-empty strings")
            frozen[key] = _freeze_json(value[key], f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _strict_json_loads(payload: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} is forbidden")

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid UTF-8 JSON metadata") from exc


def _require_exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{name} keys mismatch: missing={missing}, unknown={unknown}")


def _readonly_float_array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError(f"{name} must have dtype float32 or float64, got {value.dtype}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    dtype = np.dtype("<f4" if value.dtype.itemsize == 4 else "<f8")
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.flags.writeable = False
    return result


def _readonly_uint8_array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8, got {value.dtype}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    result = np.array(value, dtype=np.uint8, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class CanvasCropTransform:
    """Map crop-local pixel centers to a bounded full canvas."""

    canvas_width: int
    canvas_height: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int

    def __post_init__(self) -> None:
        for name in ("canvas_width", "canvas_height", "crop_width", "crop_height"):
            object.__setattr__(self, name, _strict_int(getattr(self, name), name, minimum=1))
        for name in ("crop_x", "crop_y"):
            object.__setattr__(self, name, _strict_int(getattr(self, name), name, minimum=0))
        if self.crop_x + self.crop_width > self.canvas_width:
            raise ValueError("crop extends beyond canvas width")
        if self.crop_y + self.crop_height > self.canvas_height:
            raise ValueError("crop extends beyond canvas height")

    def to_record(self) -> dict[str, int]:
        return {
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "crop_x": self.crop_x,
            "crop_y": self.crop_y,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
        }

    @classmethod
    def from_record(cls, value: object) -> "CanvasCropTransform":
        if not isinstance(value, Mapping):
            raise TypeError("canvas_crop must be an object")
        expected = {
            "canvas_width", "canvas_height", "crop_x", "crop_y", "crop_width", "crop_height"
        }
        _require_exact_keys(value, expected, "canvas_crop")
        return cls(**{key: value[key] for key in expected})

    def crop_to_canvas(self, points_xy: np.ndarray) -> np.ndarray:
        points = _points_array(points_xy)
        return points + np.asarray([self.crop_x, self.crop_y], dtype=np.float64)

    def canvas_to_crop(self, points_xy: np.ndarray) -> np.ndarray:
        points = _points_array(points_xy)
        return points - np.asarray([self.crop_x, self.crop_y], dtype=np.float64)


@dataclass(frozen=True)
class SupportSemantics:
    """Finite-support and tail-fade convention for the peak-one Gaussian kernel."""

    mode: str = "axis_aligned_bbox"
    sigma_cutoff: float = 3.0
    fade_alpha: float = 0.0
    center_rounding: str = "ties_to_even"
    minimum_radius_px: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _expect_choice(
                self.mode,
                "support.mode",
                frozenset({"axis_aligned_bbox", "ellipse", "infinite"}),
            ),
        )
        cutoff = _finite_float(self.sigma_cutoff, "support.sigma_cutoff", minimum=0.0)
        if cutoff == 0.0:
            raise ValueError("support.sigma_cutoff must be > 0")
        object.__setattr__(self, "sigma_cutoff", cutoff)
        fade = _finite_float(self.fade_alpha, "support.fade_alpha", minimum=0.0)
        if fade > 1.0:
            raise ValueError("support.fade_alpha must be <= 1")
        if self.mode == "infinite" and fade != 0.0:
            raise ValueError("infinite support cannot use cutoff tail fading")
        object.__setattr__(self, "fade_alpha", fade)
        if self.center_rounding != "ties_to_even":
            raise ValueError("support.center_rounding must be 'ties_to_even'")
        radius = _strict_int(self.minimum_radius_px, "support.minimum_radius_px", minimum=1)
        object.__setattr__(self, "minimum_radius_px", radius)
        if self.mode == "infinite" and cutoff != 3.0:
            raise ValueError("infinite support requires canonical unused sigma_cutoff=3.0")
        if self.mode != "axis_aligned_bbox" and radius != 1:
            raise ValueError(
                "non-AABB support requires canonical unused minimum_radius_px=1"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "sigma_cutoff": self.sigma_cutoff,
            "fade_alpha": self.fade_alpha,
            "center_rounding": self.center_rounding,
            "minimum_radius_px": self.minimum_radius_px,
        }

    @classmethod
    def from_record(cls, value: object) -> "SupportSemantics":
        if not isinstance(value, Mapping):
            raise TypeError("support semantics must be an object")
        expected = {"mode", "sigma_cutoff", "fade_alpha", "center_rounding", "minimum_radius_px"}
        _require_exact_keys(value, expected, "support")
        return cls(**{key: value[key] for key in expected})


@dataclass(frozen=True)
class FilterSemantics:
    """Covariance-space filtering; variance is added before inversion."""

    mode: str = "none"
    aa_dilation_px2: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _expect_choice(
                self.mode,
                "filter.mode",
                frozenset({"none", "isotropic_covariance_add"}),
            ),
        )
        dilation = _finite_float(
            self.aa_dilation_px2, "filter.aa_dilation_px2", minimum=0.0
        )
        if self.mode == "none" and dilation != 0.0:
            raise ValueError("filter.mode='none' requires zero aa_dilation_px2")
        object.__setattr__(self, "aa_dilation_px2", dilation)

    def to_record(self) -> dict[str, object]:
        return {"mode": self.mode, "aa_dilation_px2": self.aa_dilation_px2}

    @classmethod
    def from_record(cls, value: object) -> "FilterSemantics":
        if not isinstance(value, Mapping):
            raise TypeError("filter semantics must be an object")
        expected = {"mode", "aa_dilation_px2"}
        _require_exact_keys(value, expected, "filter")
        return cls(**{key: value[key] for key in expected})


@dataclass(frozen=True)
class AlphaSemantics:
    """Packed-alpha provenance, boundary policy, and optional output matting."""

    payload_encoding: str = "none"
    source_threshold: float | None = None
    matting_mode: str = "none"
    boundary_policy: str = "unconstrained"
    sampling: str = "nearest_pixel_center_half_up"

    def __post_init__(self) -> None:
        encoding = _expect_choice(
            self.payload_encoding,
            "alpha.payload_encoding",
            frozenset(
                {
                    "none",
                    "binary_exact_packbits_little",
                    "binary_thresholded_packbits_little",
                }
            ),
        )
        object.__setattr__(self, "payload_encoding", encoding)
        object.__setattr__(
            self,
            "matting_mode",
            _expect_choice(
                self.matting_mode,
                "alpha.matting_mode",
                frozenset({"none", "multiply_alpha"}),
            ),
        )
        object.__setattr__(
            self,
            "boundary_policy",
            _expect_choice(
                self.boundary_policy,
                "alpha.boundary_policy",
                frozenset({"unconstrained", "hard_contained"}),
            ),
        )
        if self.sampling != "nearest_pixel_center_half_up":
            raise ValueError("alpha.sampling must be 'nearest_pixel_center_half_up'")
        if encoding == "binary_thresholded_packbits_little":
            if self.source_threshold is None:
                raise ValueError("thresholded alpha requires source_threshold")
            threshold = _finite_float(self.source_threshold, "alpha.source_threshold")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("alpha.source_threshold must be in [0, 1]")
            object.__setattr__(self, "source_threshold", threshold)
        elif self.source_threshold is not None:
            raise ValueError("alpha.source_threshold is valid only for thresholded alpha")
        if self.matting_mode == "multiply_alpha" and encoding == "none":
            raise ValueError("alpha matting requires a packed alpha payload")

    def to_record(self) -> dict[str, object]:
        return {
            "payload_encoding": self.payload_encoding,
            "source_threshold": self.source_threshold,
            "matting_mode": self.matting_mode,
            "boundary_policy": self.boundary_policy,
            "sampling": self.sampling,
        }

    @classmethod
    def from_record(cls, value: object) -> "AlphaSemantics":
        if not isinstance(value, Mapping):
            raise TypeError("alpha semantics must be an object")
        expected = {
            "payload_encoding", "source_threshold", "matting_mode", "boundary_policy", "sampling"
        }
        _require_exact_keys(value, expected, "alpha")
        return cls(**{key: value[key] for key in expected})


@dataclass(frozen=True)
class FieldSemantics:
    """All choices that can change pixels or downstream component meaning."""

    coefficient_domain: str
    background_mode: str = "zero_dc"
    renderer_equation: str = "additive_rgb_peak_one_v1"
    structural_equation: str = "independent_additive_mass_v1"
    geometry_parameterization: str = "rs_log_scale_crop_xy_v1"
    appearance_space: str = "linear_rgb_unclipped"
    support: SupportSemantics = field(default_factory=SupportSemantics)
    filtering: FilterSemantics = field(default_factory=FilterSemantics)
    alpha: AlphaSemantics = field(default_factory=AlphaSemantics)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficient_domain",
            _expect_choice(
                self.coefficient_domain,
                "coefficient_domain",
                frozenset({"nonnegative", "signed"}),
            ),
        )
        object.__setattr__(
            self,
            "background_mode",
            _expect_choice(
                self.background_mode,
                "background_mode",
                frozenset({"zero_dc", "explicit_counted_dc"}),
            ),
        )
        constants = {
            "renderer_equation": "additive_rgb_peak_one_v1",
            "structural_equation": "independent_additive_mass_v1",
            "geometry_parameterization": "rs_log_scale_crop_xy_v1",
            "appearance_space": "linear_rgb_unclipped",
        }
        for name, expected in constants.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected!r} for schema {SCHEMA_VERSION}")
        if not isinstance(self.support, SupportSemantics):
            raise TypeError("support must be SupportSemantics")
        if not isinstance(self.filtering, FilterSemantics):
            raise TypeError("filtering must be FilterSemantics")
        if not isinstance(self.alpha, AlphaSemantics):
            raise TypeError("alpha must be AlphaSemantics")

    def to_record(self) -> dict[str, object]:
        return {
            "coefficient_domain": self.coefficient_domain,
            "background_mode": self.background_mode,
            "renderer_equation": self.renderer_equation,
            "structural_equation": self.structural_equation,
            "geometry_parameterization": self.geometry_parameterization,
            "appearance_space": self.appearance_space,
            "support": self.support.to_record(),
            "filtering": self.filtering.to_record(),
            "alpha": self.alpha.to_record(),
        }

    @classmethod
    def from_record(cls, value: object) -> "FieldSemantics":
        if not isinstance(value, Mapping):
            raise TypeError("field semantics must be an object")
        expected = {
            "coefficient_domain",
            "background_mode",
            "renderer_equation",
            "structural_equation",
            "geometry_parameterization",
            "appearance_space",
            "support",
            "filtering",
            "alpha",
        }
        _require_exact_keys(value, expected, "semantics")
        kwargs = {key: value[key] for key in expected}
        kwargs["support"] = SupportSemantics.from_record(value["support"])
        kwargs["filtering"] = FilterSemantics.from_record(value["filtering"])
        kwargs["alpha"] = AlphaSemantics.from_record(value["alpha"])
        return cls(**kwargs)


@dataclass(frozen=True)
class CameraMetadata:
    """Version-labelled, immutable JSON camera payload owned by its source application."""

    schema: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.schema, str) or not self.schema.strip():
            raise ValueError("camera.schema must be a non-empty string")
        if not isinstance(self.payload, Mapping):
            raise TypeError("camera.payload must be a mapping")
        object.__setattr__(self, "payload", _freeze_json(self.payload))

    def to_record(self) -> dict[str, object]:
        return {"schema": self.schema, "payload": _thaw_json(self.payload)}

    @classmethod
    def from_record(cls, value: object) -> "CameraMetadata":
        if not isinstance(value, Mapping):
            raise TypeError("camera metadata must be an object")
        _require_exact_keys(value, {"schema", "payload"}, "camera")
        return cls(schema=value["schema"], payload=value["payload"])


def pack_alpha(alpha: np.ndarray, *, threshold: float | None = None) -> np.ndarray:
    """Pack a crop mask in row-major order using explicit little-bit order.

    Without ``threshold``, only bool or exact {0,1} arrays are accepted.  Supplying a threshold
    is an explicitly lossy conversion and must be paired with thresholded alpha semantics.
    """
    if not isinstance(alpha, np.ndarray) or alpha.ndim != 2:
        raise TypeError("alpha must be a two-dimensional NumPy array")
    if threshold is None:
        if alpha.dtype == np.bool_:
            mask = alpha
        else:
            if not np.issubdtype(alpha.dtype, np.number) or not np.isfinite(alpha).all():
                raise ValueError("exact alpha must be finite bool or numeric {0,1}")
            if not np.logical_or(alpha == 0, alpha == 1).all():
                raise ValueError("non-binary alpha requires an explicit lossy threshold")
            mask = alpha.astype(bool)
    else:
        threshold_value = _finite_float(threshold, "threshold")
        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if not np.issubdtype(alpha.dtype, np.number) or not np.isfinite(alpha).all():
            raise ValueError("thresholded alpha must be finite numeric data")
        if ((alpha < 0.0) | (alpha > 1.0)).any():
            raise ValueError("thresholded alpha must lie in [0, 1]")
        mask = alpha >= threshold_value
    return np.packbits(np.asarray(mask, dtype=np.uint8).reshape(-1), bitorder="little")


def unpack_alpha(packed_alpha: np.ndarray, height: int, width: int) -> np.ndarray:
    height = _strict_int(height, "height", minimum=1)
    width = _strict_int(width, "width", minimum=1)
    needed = (height * width + 7) // 8
    packed = _readonly_uint8_array(packed_alpha, "packed_alpha", (needed,))
    bits = np.unpackbits(packed, bitorder="little", count=height * width)
    return bits.reshape(height, width).astype(bool, copy=False)


def _points_array(points_xy: object) -> np.ndarray:
    points = np.asarray(points_xy)
    if points.dtype.kind not in "fiu" or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have numeric shape (M, 2)")
    result = np.asarray(points, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("points_xy must contain only finite values")
    return result


@dataclass(frozen=True)
class ObservationField2D:
    """Immutable semantic field with a lossless reference representation."""

    means_xy: np.ndarray
    log_scales_xy: np.ndarray
    rotations_rad: np.ndarray
    rgb_coeff: np.ndarray
    canvas_crop: CanvasCropTransform
    semantics: FieldSemantics
    structural_mass: np.ndarray | None = None
    filter_variance_px2: np.ndarray | None = None
    background_rgb: np.ndarray | None = None
    packed_alpha: np.ndarray | None = None
    camera: CameraMetadata | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ObservationField2D schema {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        if not isinstance(self.canvas_crop, CanvasCropTransform):
            raise TypeError("canvas_crop must be CanvasCropTransform")
        if not isinstance(self.semantics, FieldSemantics):
            raise TypeError("semantics must be FieldSemantics")
        if self.camera is not None and not isinstance(self.camera, CameraMetadata):
            raise TypeError("camera must be CameraMetadata or None")

        if not isinstance(self.means_xy, np.ndarray) or self.means_xy.ndim != 2:
            raise ValueError("means_xy must have shape (N, 2)")
        n = self.means_xy.shape[0]
        arrays = {
            "means_xy": _readonly_float_array(self.means_xy, "means_xy", (n, 2)),
            "log_scales_xy": _readonly_float_array(
                self.log_scales_xy, "log_scales_xy", (n, 2)
            ),
            "rotations_rad": _readonly_float_array(
                self.rotations_rad, "rotations_rad", (n,)
            ),
            "rgb_coeff": _readonly_float_array(self.rgb_coeff, "rgb_coeff", (n, 3)),
        }
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scale_variance = np.exp(2.0 * arrays["log_scales_xy"].astype(np.float64))
        if not np.isfinite(scale_variance).all() or (scale_variance <= 0.0).any():
            raise ValueError("log_scales_xy must produce finite positive covariance")
        if self.semantics.coefficient_domain == "nonnegative" and (
            arrays["rgb_coeff"] < 0.0
        ).any():
            raise ValueError("nonnegative coefficient domain forbids negative rgb_coeff")

        if self.structural_mass is not None:
            mass = _readonly_float_array(self.structural_mass, "structural_mass", (n,))
            if (mass < 0.0).any():
                raise ValueError("structural_mass must be nonnegative")
            object.__setattr__(self, "structural_mass", mass)

        filtering = self.semantics.filtering
        if self.filter_variance_px2 is not None:
            variance = _readonly_float_array(
                self.filter_variance_px2, "filter_variance_px2", (n,)
            )
            if (variance < 0.0).any():
                raise ValueError("filter_variance_px2 must be nonnegative")
            object.__setattr__(self, "filter_variance_px2", variance)
            if filtering.mode != "isotropic_covariance_add":
                raise ValueError(
                    "filter_variance_px2 requires filter.mode='isotropic_covariance_add'"
                )
        if filtering.mode == "isotropic_covariance_add":
            if self.filter_variance_px2 is None and filtering.aa_dilation_px2 == 0.0:
                raise ValueError("isotropic covariance filtering requires nonzero filter data")

        background = self.background_rgb
        if background is not None:
            background = _readonly_float_array(background, "background_rgb", (3,))
            object.__setattr__(self, "background_rgb", background)
        expected_background = "explicit_counted_dc" if background is not None else "zero_dc"
        if self.semantics.background_mode != expected_background:
            raise ValueError(
                "background_mode does not match presence of background_rgb: "
                f"expected {expected_background!r}"
            )
        if (
            background is not None
            and self.semantics.coefficient_domain == "nonnegative"
            and (background < 0.0).any()
        ):
            raise ValueError("nonnegative coefficient domain forbids negative background_rgb")

        pixels = self.canvas_crop.crop_height * self.canvas_crop.crop_width
        alpha_bytes = (pixels + 7) // 8
        encoding = self.semantics.alpha.payload_encoding
        if self.packed_alpha is None:
            if encoding != "none":
                raise ValueError("alpha payload encoding requires packed_alpha")
        else:
            packed = _readonly_uint8_array(self.packed_alpha, "packed_alpha", (alpha_bytes,))
            if encoding == "none":
                raise ValueError("packed_alpha requires a non-'none' alpha payload encoding")
            remainder = pixels % 8
            if remainder:
                used_mask = (1 << remainder) - 1
                if int(packed[-1]) & ~used_mask:
                    raise ValueError("packed_alpha has nonzero unused tail bits")
            object.__setattr__(self, "packed_alpha", packed)

    @property
    def n(self) -> int:
        return int(self.means_xy.shape[0])

    @property
    def crop_shape(self) -> tuple[int, int]:
        return self.canvas_crop.crop_height, self.canvas_crop.crop_width

    def _array_items(self) -> dict[str, np.ndarray]:
        arrays = {
            "means_xy": self.means_xy,
            "log_scales_xy": self.log_scales_xy,
            "rotations_rad": self.rotations_rad,
            "rgb_coeff": self.rgb_coeff,
        }
        for name in sorted(_OPTIONAL_ARRAY_NAMES):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        return arrays

    def semantic_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canvas_crop": self.canvas_crop.to_record(),
            "semantics": self.semantics.to_record(),
            "camera": None if self.camera is None else self.camera.to_record(),
            "array_names": sorted(self._array_items()),
        }

    def canonical_hash(self) -> str:
        digest = hashlib.sha256(_HASH_DOMAIN)
        semantic = _canonical_json(self.semantic_record())
        digest.update(len(semantic).to_bytes(8, "little"))
        digest.update(semantic)
        for name, array in sorted(self._array_items().items()):
            encoded_name = name.encode("ascii")
            descriptor = _canonical_json(
                {"dtype": array.dtype.str, "shape": list(array.shape)}
            )
            raw = np.ascontiguousarray(array).tobytes(order="C")
            digest.update(len(encoded_name).to_bytes(4, "little"))
            digest.update(encoded_name)
            digest.update(len(descriptor).to_bytes(8, "little"))
            digest.update(descriptor)
            digest.update(len(raw).to_bytes(8, "little"))
            digest.update(raw)
        return digest.hexdigest()

    def alpha_mask(self) -> np.ndarray:
        if self.packed_alpha is None:
            raise ValueError("field has no packed alpha payload")
        return unpack_alpha(self.packed_alpha, *self.crop_shape)

    def _crop_points(self, points_xy: object, coordinate_space: CoordinateSpace) -> np.ndarray:
        points = _points_array(points_xy)
        if coordinate_space == "crop":
            return points
        if coordinate_space == "canvas":
            return self.canvas_crop.canvas_to_crop(points)
        raise ValueError("coordinate_space must be 'crop' or 'canvas'")

    def _effective_variance(self) -> tuple[np.ndarray, np.ndarray]:
        # Preserve the geometry array precision here. In particular, a float32 legacy adapter
        # must make its integer support-radius decision in the same precision as GaussianField.
        variances = np.exp(2.0 * self.log_scales_xy)
        extra = self.semantics.filtering.aa_dilation_px2
        if self.filter_variance_px2 is not None:
            extra = self.filter_variance_px2.astype(np.float64)[:, None] + extra
        variances = variances + extra
        return variances[:, 0], variances[:, 1]

    def kernel_weights(
        self, points_xy: object, *, coordinate_space: CoordinateSpace = "crop"
    ) -> np.ndarray:
        """Return peak-one kernel weights with shape ``(M, N)``."""
        points = self._crop_points(points_xy, coordinate_space)
        if self.n == 0:
            return np.zeros((points.shape[0], 0), dtype=np.float64)
        native_sx2, native_sy2 = self._effective_variance()
        native_theta = self.rotations_rad
        native_cosine = np.cos(native_theta)
        native_sine = np.sin(native_theta)
        sx2 = native_sx2.astype(np.float64)
        sy2 = native_sy2.astype(np.float64)
        theta = native_theta.astype(np.float64)
        cosine = np.cos(theta)
        sine = np.sin(theta)
        inv_sx2 = 1.0 / sx2
        inv_sy2 = 1.0 / sy2
        conic_a = cosine * cosine * inv_sx2 + sine * sine * inv_sy2
        conic_b = cosine * sine * (inv_sx2 - inv_sy2)
        conic_c = sine * sine * inv_sx2 + cosine * cosine * inv_sy2
        dx = points[:, None, 0] - self.means_xy[None, :, 0]
        dy = points[:, None, 1] - self.means_xy[None, :, 1]
        q = conic_a[None, :] * dx * dx
        q += 2.0 * conic_b[None, :] * dx * dy
        q += conic_c[None, :] * dy * dy
        weights = np.exp(-0.5 * q)

        support = self.semantics.support
        if support.mode == "ellipse":
            active = q <= support.sigma_cutoff * support.sigma_cutoff
            weights = np.where(active, weights, 0.0)
        elif support.mode == "axis_aligned_bbox":
            var_x = (
                native_cosine * native_cosine * native_sx2
                + native_sine * native_sine * native_sy2
            )
            var_y = (
                native_sine * native_sine * native_sx2
                + native_cosine * native_cosine * native_sy2
            )
            radii = np.ceil(
                support.sigma_cutoff * np.sqrt(np.stack([var_x, var_y], axis=1))
            )
            radii = np.maximum(radii, support.minimum_radius_px)
            centers = np.rint(self.means_xy.astype(np.float64))
            active = (
                (points[:, None, 0] >= centers[None, :, 0] - radii[None, :, 0])
                & (points[:, None, 0] <= centers[None, :, 0] + radii[None, :, 0])
                & (points[:, None, 1] >= centers[None, :, 1] - radii[None, :, 1])
                & (points[:, None, 1] <= centers[None, :, 1] + radii[None, :, 1])
            )
            weights = np.where(active, weights, 0.0)
        if support.fade_alpha > 0.0:
            tail = support.fade_alpha * math.exp(-0.5 * support.sigma_cutoff**2)
            weights = np.maximum(weights - tail, 0.0)
        return weights

    def appearance_raw(
        self, points_xy: object, *, coordinate_space: CoordinateSpace = "crop"
    ) -> np.ndarray:
        """Evaluate unclipped, unmatted linear RGB.  This is the appearance oracle."""
        weights = self.kernel_weights(points_xy, coordinate_space=coordinate_space)
        result = weights @ self.rgb_coeff.astype(np.float64)
        if self.background_rgb is not None:
            result = result + self.background_rgb.astype(np.float64)[None, :]
        return result

    def alpha_at(
        self, points_xy: object, *, coordinate_space: CoordinateSpace = "crop"
    ) -> np.ndarray:
        """Nearest-pixel lookup of packed binary alpha; outside the crop is zero."""
        if self.packed_alpha is None:
            raise ValueError("field has no packed alpha payload")
        points = self._crop_points(points_xy, coordinate_space)
        height, width = self.crop_shape
        ix = np.floor(points[:, 0] + 0.5).astype(np.int64)
        iy = np.floor(points[:, 1] + 0.5).astype(np.int64)
        inside = (
            (points[:, 0] >= -0.5)
            & (points[:, 0] < width - 0.5)
            & (points[:, 1] >= -0.5)
            & (points[:, 1] < height - 0.5)
        )
        alpha = np.zeros(points.shape[0], dtype=np.float64)
        mask = self.alpha_mask()
        alpha[inside] = mask[iy[inside], ix[inside]].astype(np.float64)
        return alpha

    def appearance_matted(
        self, points_xy: object, *, coordinate_space: CoordinateSpace = "crop"
    ) -> np.ndarray:
        """Explicitly multiply raw appearance by packed alpha."""
        raw = self.appearance_raw(points_xy, coordinate_space=coordinate_space)
        alpha = self.alpha_at(points_xy, coordinate_space=coordinate_space)
        return raw * alpha[:, None]

    def structural_density(
        self,
        points_xy: object,
        *,
        coordinate_space: CoordinateSpace = "crop",
        apply_alpha: bool = False,
    ) -> np.ndarray:
        if self.structural_mass is None:
            raise ValueError("field has no independently defined structural_mass")
        weights = self.kernel_weights(points_xy, coordinate_space=coordinate_space)
        density = weights @ self.structural_mass.astype(np.float64)
        if apply_alpha:
            density = density * self.alpha_at(points_xy, coordinate_space=coordinate_space)
        return density

    def responsibilities(
        self,
        points_xy: object,
        *,
        epsilon: float = 1e-8,
        coordinate_space: CoordinateSpace = "crop",
        apply_alpha: bool = False,
    ) -> np.ndarray:
        if self.structural_mass is None:
            raise ValueError("field has no independently defined structural_mass")
        epsilon_value = _finite_float(epsilon, "epsilon", minimum=0.0)
        if epsilon_value == 0.0:
            raise ValueError("epsilon must be > 0")
        weighted = self.kernel_weights(
            points_xy, coordinate_space=coordinate_space
        ) * self.structural_mass.astype(np.float64)[None, :]
        denominator = weighted.sum(axis=1, keepdims=True) + epsilon_value
        result = weighted / denominator
        if apply_alpha:
            result *= self.alpha_at(points_xy, coordinate_space=coordinate_space)[:, None]
        return result

    def _crop_grid(self) -> np.ndarray:
        height, width = self.crop_shape
        yy, xx = np.mgrid[0:height, 0:width]
        return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float64)

    def render_raw(self) -> np.ndarray:
        height, width = self.crop_shape
        return self.appearance_raw(self._crop_grid()).reshape(height, width, 3)

    def render_matted(self) -> np.ndarray:
        height, width = self.crop_shape
        return self.appearance_matted(self._crop_grid()).reshape(height, width, 3)

    def render_structural_density(self, *, apply_alpha: bool = False) -> np.ndarray:
        height, width = self.crop_shape
        return self.structural_density(
            self._crop_grid(), apply_alpha=apply_alpha
        ).reshape(height, width)

    def save_lossless(self, path: str | os.PathLike[str]) -> None:
        save_observation_field(self, path)

    @classmethod
    def load_lossless(cls, path: str | os.PathLike[str]) -> "ObservationField2D":
        return load_observation_field(path)


def clip_for_display(rgb: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Explicit display-only clipping; never called by an appearance oracle."""
    low_value = _finite_float(low, "low")
    high_value = _finite_float(high, "high")
    if low_value >= high_value:
        raise ValueError("display clip low must be smaller than high")
    array = np.asarray(rgb)
    if array.dtype.kind != "f" or array.shape[-1:] != (3,) or not np.isfinite(array).all():
        raise ValueError("rgb must be a finite floating array with trailing dimension 3")
    return np.clip(array, low_value, high_value)


def _array_descriptor(array: np.ndarray) -> dict[str, object]:
    digest = hashlib.sha256(_ARRAY_HASH_DOMAIN)
    header = _canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)})
    raw = np.ascontiguousarray(array).tobytes(order="C")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def _storage_metadata(field_value: ObservationField2D) -> dict[str, object]:
    return {
        "format": REFERENCE_CONTAINER,
        "container_version": REFERENCE_CONTAINER_VERSION,
        "field": field_value.semantic_record(),
        "arrays": {
            name: _array_descriptor(array)
            for name, array in sorted(field_value._array_items().items())
        },
        "content_sha256": field_value.canonical_hash(),
    }


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in sorted(arrays.items()):
            stream = io.BytesIO()
            np.lib.format.write_array(
                stream, np.ascontiguousarray(array), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, stream.getvalue(), compress_type=zipfile.ZIP_STORED)


def save_observation_field(
    field_value: ObservationField2D, path: str | os.PathLike[str]
) -> None:
    if not isinstance(field_value, ObservationField2D):
        raise TypeError("field_value must be ObservationField2D")
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"parent directory does not exist: {target.parent}")
    metadata = np.frombuffer(_canonical_json(_storage_metadata(field_value)), dtype=np.uint8).copy()
    arrays = {"__metadata__": metadata, **field_value._array_items()}
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_deterministic_npz(temporary, arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_archive_members(path: Path) -> set[str]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = [info.filename for info in archive.infolist()]
            if len(members) != len(set(members)):
                raise ValueError("reference field archive contains duplicate members")
            if any(info.flag_bits & 0x1 for info in archive.infolist()):
                raise ValueError("encrypted reference field members are forbidden")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid reference field NPZ container") from exc
    names: set[str] = set()
    for member in members:
        if not member.endswith(".npy") or "/" in member or "\\" in member:
            raise ValueError(f"invalid reference field member {member!r}")
        names.add(member[:-4])
    return names


def _load_array(archive: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    try:
        value = archive[name]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"cannot load array {name!r}") from exc
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise ValueError(f"array {name!r} must be a non-object NumPy array")
    return value


def load_observation_field(path: str | os.PathLike[str]) -> ObservationField2D:
    source = Path(path)
    member_names = _validate_archive_members(source)
    if "__metadata__" not in member_names:
        raise ValueError("reference field archive is missing __metadata__")
    try:
        archive_context = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read reference field NPZ arrays") from exc
    with archive_context as archive:
        metadata_array = _load_array(archive, "__metadata__")
        if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
            raise ValueError("__metadata__ must be a one-dimensional uint8 array")
        metadata = _strict_json_loads(metadata_array.tobytes())
        if not isinstance(metadata, Mapping):
            raise TypeError("reference field metadata must be an object")
        _require_exact_keys(
            metadata,
            {"format", "container_version", "field", "arrays", "content_sha256"},
            "reference field metadata",
        )
        if metadata["format"] != REFERENCE_CONTAINER:
            raise ValueError(f"unknown reference field format {metadata['format']!r}")
        if metadata["container_version"] != REFERENCE_CONTAINER_VERSION:
            raise ValueError(
                f"unsupported reference container version {metadata['container_version']!r}"
            )
        field_record = metadata["field"]
        if not isinstance(field_record, Mapping):
            raise TypeError("field metadata must be an object")
        _require_exact_keys(
            field_record,
            {"schema_version", "canvas_crop", "semantics", "camera", "array_names"},
            "field metadata",
        )
        if field_record["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported ObservationField2D schema {field_record['schema_version']!r}")
        array_names = field_record["array_names"]
        if (
            not isinstance(array_names, list)
            or not all(isinstance(name, str) for name in array_names)
            or array_names != sorted(set(array_names))
        ):
            raise ValueError("field array_names must be a sorted unique string list")
        declared_names = set(array_names)
        if not _BASE_ARRAY_NAMES.issubset(declared_names) or not declared_names <= _ALL_ARRAY_NAMES:
            raise ValueError("field array_names contain missing base or unknown arrays")
        expected_members = declared_names | {"__metadata__"}
        if member_names != expected_members:
            raise ValueError(
                "reference archive members do not match metadata: "
                f"expected={sorted(expected_members)}, actual={sorted(member_names)}"
            )
        descriptors = metadata["arrays"]
        if not isinstance(descriptors, Mapping) or set(descriptors) != declared_names:
            raise ValueError("array descriptors do not match field array_names")
        arrays: dict[str, np.ndarray] = {}
        for name in sorted(declared_names):
            array = _load_array(archive, name)
            descriptor_value = descriptors[name]
            if not isinstance(descriptor_value, Mapping):
                raise TypeError(f"array descriptor {name!r} must be an object")
            _require_exact_keys(descriptor_value, {"dtype", "shape", "sha256"}, name)
            observed_descriptor = _array_descriptor(array)
            if dict(descriptor_value) != observed_descriptor:
                raise ValueError(f"array descriptor or payload mismatch for {name!r}")
            arrays[name] = array

        canvas = CanvasCropTransform.from_record(field_record["canvas_crop"])
        semantics = FieldSemantics.from_record(field_record["semantics"])
        camera_value = field_record["camera"]
        camera = None if camera_value is None else CameraMetadata.from_record(camera_value)
        kwargs: dict[str, object] = {
            "means_xy": arrays["means_xy"],
            "log_scales_xy": arrays["log_scales_xy"],
            "rotations_rad": arrays["rotations_rad"],
            "rgb_coeff": arrays["rgb_coeff"],
            "canvas_crop": canvas,
            "semantics": semantics,
            "camera": camera,
            "schema_version": field_record["schema_version"],
        }
        for name in sorted(_OPTIONAL_ARRAY_NAMES):
            if name in arrays:
                kwargs[name] = arrays[name]
        field_value = ObservationField2D(**kwargs)
        expected_metadata = _storage_metadata(field_value)
        if _canonical_json(metadata) != _canonical_json(expected_metadata):
            raise ValueError("reference field metadata does not match decoded object")
        return field_value


@dataclass(frozen=True)
class FieldAdaptation:
    """Auditable declaration attached to every legacy/direct adapter."""

    adapter: str
    source_semantics: str
    field: ObservationField2D | None
    pixel_exact: bool
    component_semantics_exact: bool
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, str) or not self.adapter:
            raise ValueError("adapter must be a non-empty string")
        object.__setattr__(
            self,
            "source_semantics",
            _expect_choice(
                self.source_semantics,
                "source_semantics",
                frozenset(
                    {
                        "authoritative_additive_rgb_coeff_v1",
                        "factorized_additive_color_times_opacity_v1",
                        "normalized_weighted_sum_v1",
                    }
                ),
            ),
        )
        if self.field is not None and not isinstance(self.field, ObservationField2D):
            raise TypeError("field must be ObservationField2D or None")
        if isinstance(self.assumptions, (str, bytes)):
            raise TypeError("adapter assumptions must be a sequence of strings")
        assumptions = tuple(self.assumptions)
        if not assumptions or not all(isinstance(item, str) and item for item in assumptions):
            raise ValueError("adapter assumptions must be non-empty strings")
        object.__setattr__(self, "assumptions", assumptions)
        if not isinstance(self.pixel_exact, bool):
            raise TypeError("pixel_exact must be bool")
        if not isinstance(self.component_semantics_exact, bool):
            raise TypeError("component_semantics_exact must be bool")
        if self.field is None and (self.pixel_exact or self.component_semantics_exact):
            raise ValueError("an adapter without a field cannot claim exactness")
        if self.source_semantics == "normalized_weighted_sum_v1" and (
            self.pixel_exact or self.component_semantics_exact
        ):
            raise ValueError("a normalized adapter cannot claim additive semantic exactness")

    def require_pixel_exact(self) -> ObservationField2D:
        if not self.pixel_exact or self.field is None:
            raise ValueError(
                f"adapter {self.adapter!r} is not an exact additive pixel conversion"
            )
        return self.field


def adapt_direct_additive(**field_kwargs: object) -> FieldAdaptation:
    field_value = ObservationField2D(**field_kwargs)
    return FieldAdaptation(
        adapter="direct_additive_v1",
        source_semantics="authoritative_additive_rgb_coeff_v1",
        field=field_value,
        pixel_exact=True,
        component_semantics_exact=True,
        assumptions=(
            "Input rgb_coeff is already the authoritative peak-one additive coefficient.",
            "Geometry, support, filtering, alpha, crop, and camera metadata are source-bound.",
        ),
    )


def _tensor_numpy(value: object, name: str) -> np.ndarray:
    try:
        detached = value.detach()
        cpu_value = detached.cpu()
        array = cpu_value.numpy()
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise TypeError(f"legacy field {name} must be a CPU-convertible tensor") from exc
    return np.asarray(array)


def _legacy_additive_field(
    source: object,
    *,
    canvas_crop: CanvasCropTransform,
    coefficient_domain: str,
    sigma_cutoff: float,
    support_fade_alpha: float,
    aa_dilation_px2: float,
    packed_alpha: np.ndarray | None,
    alpha_semantics: AlphaSemantics | None,
    camera: CameraMetadata | None,
) -> tuple[ObservationField2D, tuple[str, ...]]:
    required = ("means", "log_scales", "rotations", "colors")
    if any(not hasattr(source, name) for name in required):
        raise TypeError("source is not a compatible GaussianField-like object")
    color_grads = getattr(source, "color_grads", None)
    if color_grads is not None:
        raise ValueError("constant-coefficient adapter cannot preserve affine color gradients")
    means = _tensor_numpy(source.means, "means")
    log_scales = _tensor_numpy(source.log_scales, "log_scales")
    rotations = _tensor_numpy(source.rotations, "rotations").reshape(-1)
    colors = _tensor_numpy(source.colors, "colors")
    opacities = getattr(source, "opacities", None)
    if opacities is None:
        opacity_values = np.ones(means.shape[0], dtype=colors.dtype)
        opacity_assumption = "Missing legacy opacity means an exact multiplicative value of one."
    else:
        try:
            opacity_values = _tensor_numpy(source.opacity_values(), "opacity_values").reshape(-1)
        except AttributeError as exc:
            raise TypeError("legacy opacity field must expose opacity_values()") from exc
        opacity_assumption = "Legacy opacity logits are converted through their sigmoid values."
    rgb_coeff = colors * opacity_values[:, None]
    legacy_filter = getattr(source, "filter_variance", None)
    filter_variance = (
        None if legacy_filter is None else _tensor_numpy(legacy_filter, "filter_variance").reshape(-1)
    )
    filter_mode = (
        "isotropic_covariance_add"
        if filter_variance is not None or float(aa_dilation_px2) != 0.0
        else "none"
    )
    alpha_value = alpha_semantics or AlphaSemantics(
        payload_encoding=("none" if packed_alpha is None else "binary_exact_packbits_little")
    )
    semantics = FieldSemantics(
        coefficient_domain=coefficient_domain,
        support=SupportSemantics(
            mode="axis_aligned_bbox",
            sigma_cutoff=sigma_cutoff,
            fade_alpha=support_fade_alpha,
        ),
        filtering=FilterSemantics(mode=filter_mode, aa_dilation_px2=aa_dilation_px2),
        alpha=alpha_value,
    )
    field_value = ObservationField2D(
        means_xy=np.asarray(means),
        log_scales_xy=np.asarray(log_scales),
        rotations_rad=np.asarray(rotations),
        rgb_coeff=np.asarray(rgb_coeff),
        canvas_crop=canvas_crop,
        semantics=semantics,
        filter_variance_px2=(None if filter_variance is None else np.asarray(filter_variance)),
        packed_alpha=packed_alpha,
        camera=camera,
    )
    assumptions = (
        "Legacy colors are constant over each component support.",
        opacity_assumption,
        "Legacy finite support is the rounded-center covariance AABB used by render.py.",
        "Legacy opacity is folded into rgb_coeff and is not relabelled structural_mass.",
    )
    return field_value, assumptions


def adapt_factorized_additive_gaussian_field(
    source: object,
    *,
    canvas_crop: CanvasCropTransform,
    coefficient_domain: str,
    sigma_cutoff: float = 3.0,
    support_fade_alpha: float = 0.0,
    aa_dilation_px2: float = 0.0,
    packed_alpha: np.ndarray | None = None,
    alpha_semantics: AlphaSemantics | None = None,
    camera: CameraMetadata | None = None,
) -> FieldAdaptation:
    field_value, assumptions = _legacy_additive_field(
        source,
        canvas_crop=canvas_crop,
        coefficient_domain=coefficient_domain,
        sigma_cutoff=sigma_cutoff,
        support_fade_alpha=support_fade_alpha,
        aa_dilation_px2=aa_dilation_px2,
        packed_alpha=packed_alpha,
        alpha_semantics=alpha_semantics,
        camera=camera,
    )
    return FieldAdaptation(
        adapter="current_factorized_additive_v1",
        source_semantics="factorized_additive_color_times_opacity_v1",
        field=field_value,
        pixel_exact=True,
        component_semantics_exact=False,
        assumptions=assumptions
        + (
            "The product preserves additive pixels but does not preserve the color/opacity gauge.",
        ),
    )


def adapt_normalized_gaussian_field(
    source: object,
    *,
    canvas_crop: CanvasCropTransform,
    coefficient_domain: str,
    permit_inexact: bool = False,
    sigma_cutoff: float = 3.0,
    support_fade_alpha: float = 0.0,
    aa_dilation_px2: float = 0.0,
    packed_alpha: np.ndarray | None = None,
    alpha_semantics: AlphaSemantics | None = None,
    camera: CameraMetadata | None = None,
) -> FieldAdaptation:
    if not isinstance(permit_inexact, bool):
        raise TypeError("permit_inexact must be bool")
    assumptions = (
        "The normalized source divides accumulated RGB by accumulated kernel weight plus epsilon.",
        "No per-row additive coefficient can preserve that spatially varying denominator in general.",
        "No normalized opacity or denominator is relabelled as independently supervised mass.",
    )
    if not permit_inexact:
        return FieldAdaptation(
            adapter="normalized_to_additive_v1",
            source_semantics="normalized_weighted_sum_v1",
            field=None,
            pixel_exact=False,
            component_semantics_exact=False,
            assumptions=assumptions
            + ("Set permit_inexact=True only for a named approximate control.",),
        )
    field_value, legacy_assumptions = _legacy_additive_field(
        source,
        canvas_crop=canvas_crop,
        coefficient_domain=coefficient_domain,
        sigma_cutoff=sigma_cutoff,
        support_fade_alpha=support_fade_alpha,
        aa_dilation_px2=aa_dilation_px2,
        packed_alpha=packed_alpha,
        alpha_semantics=alpha_semantics,
        camera=camera,
    )
    return FieldAdaptation(
        adapter="normalized_to_additive_v1",
        source_semantics="normalized_weighted_sum_v1",
        field=field_value,
        pixel_exact=False,
        component_semantics_exact=False,
        assumptions=assumptions
        + legacy_assumptions
        + ("Produced field is an explicitly inexact additive control, not a semantic conversion.",),
    )


__all__ = [
    "AlphaSemantics",
    "CameraMetadata",
    "CanvasCropTransform",
    "FieldAdaptation",
    "FieldSemantics",
    "FilterSemantics",
    "ObservationField2D",
    "REFERENCE_CONTAINER",
    "REFERENCE_CONTAINER_VERSION",
    "SCHEMA_VERSION",
    "SupportSemantics",
    "adapt_direct_additive",
    "adapt_factorized_additive_gaussian_field",
    "adapt_normalized_gaussian_field",
    "clip_for_display",
    "load_observation_field",
    "pack_alpha",
    "save_observation_field",
    "unpack_alpha",
]
