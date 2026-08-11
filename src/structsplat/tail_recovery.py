"""Default-off confidence-gated tail recovery for normalized Gaussian fields.

The ordinary normalized renderer interprets its positive denominator floor as black prior mass.
This module can replace that prior only at pixels below a declared coverage threshold with a
wider render of the same stored rows.  It is a decode-time research candidate: no field tensor is
mutated, no Gaussian is added, and every pixel outside the activation mask is bit-identical to the
ordinary render.

Torch remains a lazy dependency so importing this module preserves the package's NumPy-only
analysis boundary.
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


_NORMALIZED_RENDERERS = (
    "normalized",
    "cuda",
    "cuda_normalized",
    "cuda_tiled",
    "cuda_tiled_normalized",
)
_SST1_HEADER = struct.Struct("<4sIII")
_SST1_MAGIC = b"SST1"
_UINT32_MAX = (1 << 32) - 1


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {result}")
    return result


@dataclass(frozen=True)
class ConfidenceTailConfig:
    """Fixed decode-time geometry for the same-field confidence prior."""

    scale_multiplier: float = 2.0
    coverage_threshold: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scale_multiplier",
            _finite_positive(self.scale_multiplier, "scale_multiplier"),
        )
        if self.scale_multiplier <= 1.0:
            raise ValueError(
                f"scale_multiplier must be > 1, got {self.scale_multiplier}"
            )
        if self.coverage_threshold is not None:
            object.__setattr__(
                self,
                "coverage_threshold",
                _finite_positive(self.coverage_threshold, "coverage_threshold"),
            )


@dataclass(frozen=True)
class ConfidenceTailResult:
    """Cold baseline, wider prior, candidate, and confidence certificate tensors."""

    baseline: torch.Tensor
    prior: torch.Tensor
    recovered: torch.Tensor
    candidate: torch.Tensor
    denominator: torch.Tensor
    missing_mass: torch.Tensor
    activation_mask: torch.Tensor
    scale_multiplier: float
    coverage_threshold: float
    normalization_eps: float

    @property
    def activation_count(self) -> int:
        return int(self.activation_mask.detach().sum().item())

    @property
    def outside_identity_max_abs(self) -> float:
        inactive = ~self.activation_mask
        if not bool(inactive.any()):
            return 0.0
        delta = (self.candidate - self.baseline).abs()[inactive]
        return float(delta.max().detach().cpu()) if delta.numel() else 0.0


def _positive_uint32(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0 or value > _UINT32_MAX:
        raise ValueError(f"{name} must be in [1, {_UINT32_MAX}], got {value}")
    return value


@dataclass(frozen=True)
class SparseTailPayload:
    """Canonical SST1 raster-index sidecar for sparse recovered-pixel substitution."""

    height: int
    width: int
    flat_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        height = _positive_uint32(self.height, "height")
        width = _positive_uint32(self.width, "width")
        if not isinstance(self.flat_indices, tuple):
            raise TypeError("flat_indices must be a tuple")
        if len(self.flat_indices) > _UINT32_MAX:
            raise ValueError("flat_indices exceeds the SST1 uint32 count limit")
        pixel_count = height * width
        if len(self.flat_indices) > pixel_count:
            raise ValueError("flat_indices count exceeds the SST1 raster size")
        previous = -1
        for position, index in enumerate(self.flat_indices):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(f"flat_indices[{position}] must be an integer")
            if index < 0 or index >= pixel_count or index > _UINT32_MAX:
                raise ValueError(
                    f"flat_indices[{position}]={index} is outside the {height}x{width} raster"
                )
            if index <= previous:
                raise ValueError("flat_indices must be strictly increasing and unique")
            previous = index

    @property
    def count(self) -> int:
        return len(self.flat_indices)

    @property
    def encoded_size(self) -> int:
        return _SST1_HEADER.size + 4 * self.count

    def to_bytes(self) -> bytes:
        """Encode the canonical little-endian SST1 representation."""
        header = _SST1_HEADER.pack(_SST1_MAGIC, self.height, self.width, self.count)
        if not self.flat_indices:
            return header
        return header + struct.pack(f"<{self.count}I", *self.flat_indices)

    @classmethod
    def from_bytes(cls, payload: bytes | bytearray | memoryview) -> SparseTailPayload:
        """Parse SST1, rejecting malformed, non-canonical, or trailing input."""
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        data = bytes(payload)
        if len(data) < _SST1_HEADER.size:
            raise ValueError("SST1 payload is shorter than its 16-byte header")
        magic, height, width, count = _SST1_HEADER.unpack_from(data)
        if magic != _SST1_MAGIC:
            raise ValueError(f"invalid SST1 magic {magic!r}")
        expected = _SST1_HEADER.size + 4 * count
        if len(data) != expected:
            raise ValueError(
                f"SST1 payload length is {len(data)} bytes; expected exactly {expected}"
            )
        indices = (
            ()
            if count == 0
            else tuple(struct.unpack_from(f"<{count}I", data, _SST1_HEADER.size))
        )
        return cls(height=height, width=width, flat_indices=indices)


@dataclass(frozen=True)
class PixelSafeTailResult:
    """Encoder-selected sparse candidate and its pointwise safety certificates."""

    candidate: torch.Tensor
    selected_mask: torch.Tensor
    raw_improvement_mask: torch.Tensor
    display_nonregression_mask: torch.Tensor
    payload: SparseTailPayload

    @property
    def selected_count(self) -> int:
        return self.payload.count

    def outside_identity_max_abs(self, baseline: torch.Tensor) -> float:
        inactive = ~self.selected_mask
        if not bool(inactive.any()):
            return 0.0
        delta = (self.candidate - baseline).abs()[inactive]
        return float(delta.max().detach().cpu()) if delta.numel() else 0.0


def _validate_tail_shapes(result: ConfidenceTailResult) -> tuple[int, int]:
    baseline = result.baseline
    if baseline.ndim != 3 or baseline.shape[-1] != 3:
        raise ValueError("tail baseline must have shape (H, W, 3)")
    height, width = int(baseline.shape[0]), int(baseline.shape[1])
    for name in ("prior", "recovered", "candidate"):
        value = getattr(result, name)
        if value.shape != baseline.shape:
            raise ValueError(f"tail {name} shape must match the baseline")
        if value.device != baseline.device:
            raise ValueError(f"tail {name} device must match the baseline")
    for name in ("denominator", "missing_mass", "activation_mask"):
        value = getattr(result, name)
        if value.shape != (height, width):
            raise ValueError(f"tail {name} must have shape (H, W)")
        if value.device != baseline.device:
            raise ValueError(f"tail {name} device must match the baseline")
    return height, width


def _display_quantize(value: torch.Tensor) -> torch.Tensor:
    import torch

    return torch.round(torch.clamp(value, 0.0, 1.0) * 255.0) / 255.0


def select_pixel_safe_tail(
    tail: ConfidenceTailResult,
    target: torch.Tensor,
) -> PixelSafeTailResult:
    """Select source-known proposal sites that are safe in raw and displayed RGB error.

    The target is used only by this encoder-side selection function.  The returned SST1 payload
    contains no colors or source data; :func:`apply_sparse_tail_payload` is target-independent.
    """
    import torch

    if not isinstance(tail, ConfidenceTailResult):
        raise TypeError("tail must be a ConfidenceTailResult")
    height, width = _validate_tail_shapes(tail)
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")
    if target.shape != tail.baseline.shape:
        raise ValueError("target shape must match the tail baseline")
    if target.device != tail.baseline.device:
        raise ValueError("target device must match the tail baseline")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target must contain only finite values")
    if not bool(torch.isfinite(tail.baseline).all()) or not bool(
        torch.isfinite(tail.recovered).all()
    ):
        raise ValueError("tail baseline and recovered image must be finite")

    with torch.no_grad():
        baseline64 = tail.baseline.to(dtype=torch.float64)
        recovered64 = tail.recovered.to(dtype=torch.float64)
        target64 = target.to(dtype=torch.float64)
        baseline_raw_sse = torch.square(baseline64 - target64).sum(dim=-1)
        recovered_raw_sse = torch.square(recovered64 - target64).sum(dim=-1)
        raw_improvement = recovered_raw_sse < baseline_raw_sse

        display_target = _display_quantize(target)
        baseline_display = _display_quantize(tail.baseline)
        recovered_display = _display_quantize(tail.recovered)
        baseline_display_sse = torch.square(baseline_display - display_target).sum(dim=-1)
        recovered_display_sse = torch.square(recovered_display - display_target).sum(dim=-1)
        display_nonregression = recovered_display_sse <= baseline_display_sse

        selected = tail.activation_mask & raw_improvement & display_nonregression
        flat_indices = tuple(
            int(index)
            for index in torch.nonzero(selected.reshape(-1), as_tuple=False)
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )
        sparse_payload = SparseTailPayload(height, width, flat_indices)
        candidate = torch.where(selected.unsqueeze(-1), tail.recovered, tail.baseline)

    return PixelSafeTailResult(
        candidate=candidate,
        selected_mask=selected,
        raw_improvement_mask=raw_improvement,
        display_nonregression_mask=display_nonregression,
        payload=sparse_payload,
    )


def apply_sparse_tail_payload(
    tail: ConfidenceTailResult,
    payload: SparseTailPayload,
) -> torch.Tensor:
    """Apply explicit SST1 indices to a recomputed same-field recovered image."""
    import torch

    if not isinstance(tail, ConfidenceTailResult):
        raise TypeError("tail must be a ConfidenceTailResult")
    if not isinstance(payload, SparseTailPayload):
        raise TypeError("payload must be a SparseTailPayload")
    height, width = _validate_tail_shapes(tail)
    if (payload.height, payload.width) != (height, width):
        raise ValueError(
            "SST1 raster dimensions do not match the recomputed tail: "
            f"{payload.height}x{payload.width} != {height}x{width}"
        )
    with torch.no_grad():
        candidate = tail.baseline.clone()
        if payload.flat_indices:
            indices = torch.tensor(
                payload.flat_indices, device=candidate.device, dtype=torch.long
            )
            candidate.reshape(-1, 3)[indices] = tail.recovered.reshape(-1, 3)[indices]
    return candidate


def _render_normalized_sites(
    field: GaussianField,
    fit_config: FitConfig,
    height: int,
    width: int,
    flat_indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the normalized field and its denominator only at explicit raster sites."""
    import torch

    from .render import _support_weight

    count = len(flat_indices)
    device = field.means.device
    dtype = field.means.dtype
    if count == 0:
        return (
            torch.empty((0, 3), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=dtype),
        )

    indices = torch.tensor(flat_indices, device=device, dtype=torch.long)
    pixel_x = indices.remainder(width).to(dtype=dtype)
    pixel_y = torch.div(indices, width, rounding_mode="floor").to(dtype=dtype)
    means = field.means.detach()
    conics = field.conics(fit_config.aa_dilation).detach()
    radii = field.radii(fit_config.sigma_cutoff, fit_config.aa_dilation)
    colors = field.colors.detach()
    opacities = field.opacity_values()
    if opacities is not None:
        opacities = opacities.detach()
    rounded_x = torch.round(means[:, 0]).long()
    rounded_y = torch.round(means[:, 1]).long()
    effective_scales = field.effective_scales(0.0).detach()
    rotations = field.rotations.detach()
    color_grads = None if field.color_grads is None else field.color_grads.detach()

    # Bound the temporary (site, Gaussian) matrices independently of image dimensions.  SST1
    # payloads are normally tiny, but hostile-yet-valid payloads must not allocate K*N at once.
    site_chunk = max(1, 1_048_576 // max(int(field.n), 1))
    output: list[torch.Tensor] = []
    denominators: list[torch.Tensor] = []
    for start in range(0, count, site_chunk):
        end = min(start + site_chunk, count)
        dx = pixel_x[start:end, None] - means[None, :, 0]
        dy = pixel_y[start:end, None] - means[None, :, 1]
        inside = (
            (pixel_x[start:end, None].long() >= rounded_x[None, :] - radii[None, :, 0])
            & (pixel_x[start:end, None].long() <= rounded_x[None, :] + radii[None, :, 0])
            & (pixel_y[start:end, None].long() >= rounded_y[None, :] - radii[None, :, 1])
            & (pixel_y[start:end, None].long() <= rounded_y[None, :] + radii[None, :, 1])
        )
        a = conics[None, :, 0]
        b = conics[None, :, 1]
        c = conics[None, :, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        weights = _support_weight(
            q,
            fit_config.sigma_cutoff,
            fit_config.support_fade,
            None,
        )
        weights = torch.where(inside, weights, torch.zeros_like(weights))
        if opacities is not None:
            weights = weights * opacities[None, :]
        denominator = weights.sum(dim=1)
        if color_grads is None:
            numerator = weights @ colors
        else:
            cos_theta = torch.cos(rotations)[None, :]
            sin_theta = torch.sin(rotations)[None, :]
            scale_x = effective_scales[None, :, 0].clamp_min(1e-6)
            scale_y = effective_scales[None, :, 1].clamp_min(1e-6)
            local_x = (cos_theta * dx + sin_theta * dy) / scale_x
            local_y = (-sin_theta * dx + cos_theta * dy) / scale_y
            pixel_colors = (
                colors[None, :, :]
                + color_grads[None, :, 0, :] * local_x[:, :, None]
                + color_grads[None, :, 1, :] * local_y[:, :, None]
            )
            numerator = (weights[:, :, None] * pixel_colors).sum(dim=1)
        output.append(numerator / (denominator[:, None] + fit_config.normalization_eps))
        denominators.append(denominator)
    return torch.cat(output, dim=0), torch.cat(denominators, dim=0)


def render_sparse_tail_payload(
    field: GaussianField,
    fit_config: FitConfig,
    height: int,
    width: int,
    payload: SparseTailPayload,
    config: ConfidenceTailConfig | None = None,
) -> torch.Tensor:
    """Decode SST1 with one ordinary render and coordinate-only tail evaluation.

    Empty SST1 is the ordinary normalized renderer exactly.  For nonempty SST1, only the stored
    coordinates pay for original-denominator and twice-scale-prior evaluation; no full-frame wide
    prior is constructed.
    """
    import torch

    from .fit import _render

    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    if fit_config.renderer not in _NORMALIZED_RENDERERS:
        raise ValueError(
            "sparse tail payload requires a normalized renderer, got "
            f"{fit_config.renderer!r}"
        )
    if not isinstance(payload, SparseTailPayload):
        raise TypeError("payload must be a SparseTailPayload")
    if (payload.height, payload.width) != (height, width):
        raise ValueError(
            "SST1 raster dimensions do not match the requested render: "
            f"{payload.height}x{payload.width} != {height}x{width}"
        )
    tail_config = ConfidenceTailConfig() if config is None else config
    if not isinstance(tail_config, ConfidenceTailConfig):
        raise TypeError("config must be a ConfidenceTailConfig")
    eps = _finite_positive(fit_config.normalization_eps, "normalization_eps")

    with torch.no_grad():
        baseline = _render(field, fit_config, height, width)
        if not payload.flat_indices:
            return baseline
        _, denominator = _render_normalized_sites(
            field, fit_config, height, width, payload.flat_indices
        )
        wide_field = field.detached()
        wide_field.log_scales.add_(math.log(tail_config.scale_multiplier))
        prior, _ = _render_normalized_sites(
            wide_field, fit_config, height, width, payload.flat_indices
        )
        indices = torch.tensor(
            payload.flat_indices, device=baseline.device, dtype=torch.long
        )
        recovered = baseline.reshape(-1, 3)[indices] + (
            eps / (denominator + eps)
        ).unsqueeze(-1) * prior
        candidate = baseline.clone()
        candidate.reshape(-1, 3)[indices] = recovered
    return candidate


def render_confidence_gated_self_prior(
    field: GaussianField,
    fit_config: FitConfig,
    height: int,
    width: int,
    config: ConfidenceTailConfig | None = None,
) -> ConfidenceTailResult:
    """Render a same-row twice-scale prior only where raw normalized coverage is too low.

    If ``C0 = N / (D + eps)`` is the ordinary render and ``P`` is the wider same-field prior,
    the activated value is ``C0 + eps / (D + eps) * P``.  The second term replaces the renderer's
    implicit black prior; a final ``where`` makes non-activated pixels exactly equal to ``C0``.
    """
    import torch

    from .fit import _normalized_color_denominator, _render

    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    if fit_config.renderer not in _NORMALIZED_RENDERERS:
        raise ValueError(
            "confidence-gated self-prior requires a normalized renderer, got "
            f"{fit_config.renderer!r}"
        )
    tail_config = ConfidenceTailConfig() if config is None else config
    if not isinstance(tail_config, ConfidenceTailConfig):
        raise TypeError("config must be a ConfidenceTailConfig")
    eps = _finite_positive(fit_config.normalization_eps, "normalization_eps")
    threshold = eps if tail_config.coverage_threshold is None else float(
        tail_config.coverage_threshold
    )

    with torch.no_grad():
        baseline = _render(field, fit_config, height, width)
        denominator = _normalized_color_denominator(
            field, fit_config, height, width
        ).reshape(height, width)
        wide_field = field.detached()
        wide_field.log_scales.add_(math.log(tail_config.scale_multiplier))
        prior = _render(wide_field, fit_config, height, width)
        missing_mass = eps / (denominator + eps)
        activation_mask = denominator < threshold
        recovered = baseline + missing_mass.unsqueeze(-1) * prior
        candidate = torch.where(activation_mask.unsqueeze(-1), recovered, baseline)

    return ConfidenceTailResult(
        baseline=baseline,
        prior=prior,
        recovered=recovered,
        candidate=candidate,
        denominator=denominator,
        missing_mass=missing_mass,
        activation_mask=activation_mask,
        scale_multiplier=tail_config.scale_multiplier,
        coverage_threshold=threshold,
        normalization_eps=eps,
    )
