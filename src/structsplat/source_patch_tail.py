"""Default-off source-RGB exception patches around normalized low-coverage sites.

SPT1 is deliberately an explicit residual representation, not a Gaussian-field mutation.  The
encoder expands low-coverage seeds into coherent square neighborhoods and stores exact RGB8 values
only where the ordinary displayed render differs from the source.  Decode needs the unchanged
field plus SPT1 bytes, never the source image.

Torch remains lazy so importing the module preserves StructSplat's NumPy-only analysis boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

    from .config import FitConfig
    from .gaussians import GaussianField


_HEADER = struct.Struct("<4sIII")
_RECORD = struct.Struct("<IBBB")
_MAGIC = b"SPT1"
_UINT32_MAX = (1 << 32) - 1
_NORMALIZED_RENDERERS = (
    "normalized",
    "cuda",
    "cuda_normalized",
    "cuda_tiled",
    "cuda_tiled_normalized",
)


def _positive_uint32(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0 or value > _UINT32_MAX:
        raise ValueError(f"{name} must be in [1, {_UINT32_MAX}], got {value}")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {result}")
    return result


@dataclass(frozen=True)
class SourcePatchConfig:
    """Fixed encoder proposal geometry for coherent low-coverage patches."""

    radius: int = 3
    coverage_threshold: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.radius, bool) or not isinstance(self.radius, int):
            raise TypeError("radius must be an integer")
        if self.radius < 0:
            raise ValueError(f"radius must be >= 0, got {self.radius}")
        if self.coverage_threshold is not None:
            object.__setattr__(
                self,
                "coverage_threshold",
                _finite_positive(self.coverage_threshold, "coverage_threshold"),
            )


@dataclass(frozen=True)
class SourcePatchPayload:
    """Canonical SPT1 raster indices and exact RGB8 exception values."""

    height: int
    width: int
    flat_indices: tuple[int, ...] = ()
    rgb8: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        height = _positive_uint32(self.height, "height")
        width = _positive_uint32(self.width, "width")
        if not isinstance(self.flat_indices, tuple):
            raise TypeError("flat_indices must be a tuple")
        if not isinstance(self.rgb8, tuple):
            raise TypeError("rgb8 must be a tuple")
        if len(self.flat_indices) != len(self.rgb8):
            raise ValueError("flat_indices and rgb8 must have equal length")
        if len(self.flat_indices) > min(height * width, _UINT32_MAX):
            raise ValueError("record count exceeds the SPT1 raster/count limit")
        previous = -1
        for position, (index, color) in enumerate(zip(self.flat_indices, self.rgb8)):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(f"flat_indices[{position}] must be an integer")
            if index < 0 or index >= height * width or index > _UINT32_MAX:
                raise ValueError(
                    f"flat_indices[{position}]={index} is outside the {height}x{width} raster"
                )
            if index <= previous:
                raise ValueError("flat_indices must be strictly increasing and unique")
            previous = index
            if not isinstance(color, tuple) or len(color) != 3:
                raise TypeError(f"rgb8[{position}] must be an RGB tuple")
            for channel, value in enumerate(color):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"rgb8[{position}][{channel}] must be an integer")
                if value < 0 or value > 255:
                    raise ValueError(
                        f"rgb8[{position}][{channel}] must be in [0, 255], got {value}"
                    )

    @property
    def count(self) -> int:
        return len(self.flat_indices)

    @property
    def encoded_size(self) -> int:
        return _HEADER.size + _RECORD.size * self.count

    def to_bytes(self) -> bytes:
        encoded = bytearray(_HEADER.pack(_MAGIC, self.height, self.width, self.count))
        for index, color in zip(self.flat_indices, self.rgb8):
            encoded.extend(_RECORD.pack(index, *color))
        return bytes(encoded)

    @classmethod
    def from_bytes(cls, payload: bytes | bytearray | memoryview) -> SourcePatchPayload:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        data = bytes(payload)
        if len(data) < _HEADER.size:
            raise ValueError("SPT1 payload is shorter than its 16-byte header")
        magic, height, width, count = _HEADER.unpack_from(data)
        if magic != _MAGIC:
            raise ValueError(f"invalid SPT1 magic {magic!r}")
        expected = _HEADER.size + _RECORD.size * count
        if len(data) != expected:
            raise ValueError(
                f"SPT1 payload length is {len(data)} bytes; expected exactly {expected}"
            )
        indices: list[int] = []
        colors: list[tuple[int, int, int]] = []
        offset = _HEADER.size
        for _ in range(count):
            index, red, green, blue = _RECORD.unpack_from(data, offset)
            indices.append(index)
            colors.append((red, green, blue))
            offset += _RECORD.size
        return cls(height, width, tuple(indices), tuple(colors))


@dataclass(frozen=True)
class SourcePatchResult:
    """Encoder-selected SPT1 candidate and its local safety masks."""

    candidate: torch.Tensor
    seed_mask: torch.Tensor
    expanded_mask: torch.Tensor
    selected_mask: torch.Tensor
    payload: SourcePatchPayload
    coverage_threshold: float
    radius: int
    pointwise_raw_sse_delta_max: float
    pointwise_display_sse_delta_max: float

    @property
    def seed_count(self) -> int:
        return int(self.seed_mask.detach().sum().item())

    @property
    def expanded_count(self) -> int:
        return int(self.expanded_mask.detach().sum().item())

    @property
    def selected_count(self) -> int:
        return self.payload.count

    def outside_identity_max_abs(self, baseline: torch.Tensor) -> float:
        inactive = ~self.selected_mask
        if not bool(inactive.any()):
            return 0.0
        delta = (self.candidate - baseline).abs()[inactive]
        return float(delta.max().detach().cpu()) if delta.numel() else 0.0


def select_source_patch_tail(
    baseline: torch.Tensor,
    denominator: torch.Tensor,
    target: torch.Tensor,
    normalization_eps: float,
    config: SourcePatchConfig | None = None,
) -> SourcePatchResult:
    """Build a source-known coherent RGB8 exception candidate around coverage holes."""
    import torch
    import torch.nn.functional as functional

    if not isinstance(baseline, torch.Tensor):
        raise TypeError("baseline must be a torch.Tensor")
    if not isinstance(denominator, torch.Tensor):
        raise TypeError("denominator must be a torch.Tensor")
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")
    if baseline.ndim != 3 or baseline.shape[-1] != 3:
        raise ValueError("baseline must have shape (H, W, 3)")
    height, width = int(baseline.shape[0]), int(baseline.shape[1])
    if target.shape != baseline.shape:
        raise ValueError("target shape must match baseline")
    if denominator.shape != (height, width):
        raise ValueError("denominator must have shape (H, W)")
    if target.device != baseline.device or denominator.device != baseline.device:
        raise ValueError("baseline, denominator, and target must share a device")
    if not bool(torch.isfinite(baseline).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("baseline and target must contain only finite values")
    if not bool(torch.isfinite(denominator).all()) or bool((denominator < 0.0).any()):
        raise ValueError("denominator must be finite and nonnegative")
    if bool((target < 0.0).any()) or bool((target > 1.0).any()):
        raise ValueError("target must lie in [0, 1]")
    eps = _finite_positive(normalization_eps, "normalization_eps")
    patch_config = SourcePatchConfig() if config is None else config
    if not isinstance(patch_config, SourcePatchConfig):
        raise TypeError("config must be a SourcePatchConfig")
    threshold = (
        eps
        if patch_config.coverage_threshold is None
        else float(patch_config.coverage_threshold)
    )

    with torch.no_grad():
        seed = denominator < threshold
        if patch_config.radius == 0:
            expanded = seed.clone()
        else:
            kernel = 2 * patch_config.radius + 1
            expanded = (
                functional.max_pool2d(
                    seed.to(dtype=torch.float32)[None, None],
                    kernel_size=kernel,
                    stride=1,
                    padding=patch_config.radius,
                )[0, 0]
                > 0.0
            )
        target_rgb8 = torch.round(torch.clamp(target, 0.0, 1.0) * 255.0).to(
            dtype=torch.uint8
        )
        baseline_rgb8 = torch.round(torch.clamp(baseline, 0.0, 1.0) * 255.0).to(
            dtype=torch.uint8
        )
        decoded_target = target_rgb8.to(dtype=baseline.dtype) / 255.0
        baseline_raw_sse = torch.square(
            baseline.to(dtype=torch.float64) - target.to(dtype=torch.float64)
        ).sum(dim=-1)
        candidate_raw_sse = torch.square(
            decoded_target.to(dtype=torch.float64) - target.to(dtype=torch.float64)
        ).sum(dim=-1)
        display_changed = (baseline_rgb8 != target_rgb8).any(dim=-1)
        raw_improvement = candidate_raw_sse < baseline_raw_sse
        selected = expanded & display_changed & raw_improvement
        candidate = torch.where(selected.unsqueeze(-1), decoded_target, baseline)

        indices_tensor = (
            torch.nonzero(selected.reshape(-1), as_tuple=False).reshape(-1).detach().cpu()
        )
        flat_indices = tuple(int(index) for index in indices_tensor.tolist())
        selected_colors = target_rgb8.reshape(-1, 3)[
            indices_tensor.to(device=target_rgb8.device)
        ].detach().cpu().tolist()
        rgb8 = tuple(tuple(int(channel) for channel in color) for color in selected_colors)
        payload = SourcePatchPayload(height, width, flat_indices, rgb8)
        if selected.any():
            raw_delta_max = float(
                (candidate_raw_sse - baseline_raw_sse)[selected].max().detach().cpu()
            )
            candidate_display_sse = torch.square(
                torch.round(torch.clamp(candidate, 0.0, 1.0) * 255.0)
                - target_rgb8.to(dtype=candidate.dtype)
            ).sum(dim=-1)
            baseline_display_sse = torch.square(
                baseline_rgb8.to(dtype=candidate.dtype)
                - target_rgb8.to(dtype=candidate.dtype)
            ).sum(dim=-1)
            display_delta_max = float(
                (candidate_display_sse - baseline_display_sse)[selected]
                .max()
                .detach()
                .cpu()
            )
        else:
            raw_delta_max = 0.0
            display_delta_max = 0.0

    return SourcePatchResult(
        candidate=candidate,
        seed_mask=seed,
        expanded_mask=expanded,
        selected_mask=selected,
        payload=payload,
        coverage_threshold=threshold,
        radius=patch_config.radius,
        pointwise_raw_sse_delta_max=raw_delta_max,
        pointwise_display_sse_delta_max=display_delta_max,
    )


def apply_source_patch_payload(
    baseline: torch.Tensor,
    payload: SourcePatchPayload,
) -> torch.Tensor:
    """Apply explicit SPT1 RGB8 records to an ordinary render."""
    import torch

    if not isinstance(baseline, torch.Tensor):
        raise TypeError("baseline must be a torch.Tensor")
    if not isinstance(payload, SourcePatchPayload):
        raise TypeError("payload must be a SourcePatchPayload")
    if baseline.ndim != 3 or baseline.shape[-1] != 3:
        raise ValueError("baseline must have shape (H, W, 3)")
    height, width = int(baseline.shape[0]), int(baseline.shape[1])
    if (payload.height, payload.width) != (height, width):
        raise ValueError(
            "SPT1 raster dimensions do not match the baseline: "
            f"{payload.height}x{payload.width} != {height}x{width}"
        )
    with torch.no_grad():
        candidate = baseline.clone()
        if payload.count:
            indices = torch.tensor(
                payload.flat_indices, device=baseline.device, dtype=torch.long
            )
            colors = torch.tensor(payload.rgb8, device=baseline.device, dtype=baseline.dtype)
            candidate.reshape(-1, 3)[indices] = colors / 255.0
    return candidate


def render_source_patch_payload(
    field: GaussianField,
    fit_config: FitConfig,
    height: int,
    width: int,
    payload: SourcePatchPayload,
) -> torch.Tensor:
    """Render the unchanged normalized field once and apply SPT1 records."""
    from .fit import _render

    if fit_config.renderer not in _NORMALIZED_RENDERERS:
        raise ValueError(
            f"source patch payload requires a normalized renderer, got {fit_config.renderer!r}"
        )
    baseline = _render(field, fit_config, height, width)
    return apply_source_patch_payload(baseline, payload)
