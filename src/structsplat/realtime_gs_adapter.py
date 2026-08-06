"""Lazy realtime-gs adapter for CORE-016/ADR-0032 dual-plane packets.

Importing this module does not import torch or realtime-gs.  :func:`make_realtime_gs_view` performs
the optional imports only when a caller explicitly requests the cross-repository bridge.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .codec_native_field import CodecNativeField


def _config_digest(packet: CodecNativeField) -> str:
    payload = json.dumps(
        asdict(packet.config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _producer_source_digest() -> str:
    """Hash the exact reference producer/adapter sources, not the input photograph."""
    digest = hashlib.sha256()
    for path in sorted(
        (
            Path(__file__).resolve(),
            Path(__file__).resolve().with_name("codec_native_field.py"),
        )
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class RealtimeGSCodecNativeView:
    """Required pair for realtime-gs: structural proposals plus appearance query backend.

    The ``structural_field`` alone is intentionally not described as a faithful image teacher.
    Downstream code must pass ``query_backend`` through realtime-gs's existing pluggable backend
    seam whenever it evaluates appearance.
    """

    structural_field: Any
    query_backend: Any
    alpha_crop: Any


class CodecNativeObservationBackend:
    """Torch implementation of realtime-gs's ``ObservationQueryBackend`` protocol."""

    def __init__(
        self,
        packet: CodecNativeField,
        structural_field: Any,
        structural_backend: Any,
        *,
        device: Any,
        payload_bytes: int | None = None,
    ) -> None:
        try:
            import torch
            from rtgs.core.observation2d import ObservationQuery
        except ImportError as exc:  # pragma: no cover - exercised in an isolated subprocess test
            raise RuntimeError(
                "the codec-native realtime-gs adapter requires torch and the optional "
                "realtime-gs package"
            ) from exc
        self._torch = torch
        self._observation_query = ObservationQuery
        self.field = structural_field
        self.structural_backend = structural_backend
        self.payload_bytes = int(
            getattr(structural_backend, "payload_bytes", 0)
            if payload_bytes is None
            else payload_bytes
        )
        self.n_entries = int(getattr(structural_backend, "n_entries", 0))
        self.max_candidates = int(getattr(structural_backend, "max_candidates", 0))
        self.component_id_dtype = getattr(structural_backend, "component_id_dtype", None)
        self.total_pairs_evaluated = 0
        self.last_pair_chunk = 0
        self.peak_pair_chunk = 0

        dtype = structural_field.dtype
        self._appearance = torch.from_numpy(packet.appearance_coefficients.copy()).to(
            device=device, dtype=dtype
        )
        self._alpha = torch.from_numpy(packet.alpha_mask.copy()).to(device=device)
        crop = packet.canvas_crop
        self._origin = torch.tensor(
            [crop.crop_x + 0.5, crop.crop_y + 0.5], device=device, dtype=dtype
        )
        radius = packet.config.lattice_radius_px
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        self._offset_x = torch.from_numpy(xx.reshape(1, -1).astype(np.int64)).to(device)
        self._offset_y = torch.from_numpy(yy.reshape(1, -1).astype(np.int64)).to(device)
        self._sigma2 = float(packet.config.lattice_sigma_px**2)

    def _sync_structural_counters(self) -> None:
        """Mirror optional realtime-gs query telemetry through the paired wrapper."""
        for name in ("total_pairs_evaluated", "last_pair_chunk", "peak_pair_chunk"):
            value = getattr(self.structural_backend, name, None)
            if value is not None:
                setattr(self, name, int(value))

    def _appearance_query(self, xy: Any) -> tuple[Any, Any, Any]:
        torch = self._torch
        if not torch.is_tensor(xy) or xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must be a torch tensor with shape (S,2)")
        if not xy.is_floating_point() or not bool(torch.isfinite(xy).all()):
            raise ValueError("xy must be finite and floating-point")
        target_device = xy.device
        work_xy = xy.to(device=self._appearance.device, dtype=self._appearance.dtype)
        height, width = self._appearance.shape[:2]
        local = work_xy - self._origin
        base_x = torch.floor(local[:, 0]).to(torch.long)[:, None]
        base_y = torch.floor(local[:, 1]).to(torch.long)[:, None]
        raw_x = base_x + self._offset_x
        raw_y = base_y + self._offset_y
        present = (raw_x >= 0) & (raw_x < width) & (raw_y >= 0) & (raw_y < height)
        gather_x = raw_x.clamp(0, width - 1)
        gather_y = raw_y.clamp(0, height - 1)
        dx = local[:, None, 0] - raw_x
        dy = local[:, None, 1] - raw_y
        weights = torch.exp(-0.5 * (dx.square() + dy.square()) / self._sigma2)
        weights = weights * present.to(weights)
        denominator = weights.sum(dim=1)
        valid = (
            (local[:, 0] >= -0.5)
            & (local[:, 0] < width - 0.5)
            & (local[:, 1] >= -0.5)
            & (local[:, 1] < height - 0.5)
            & (denominator > torch.finfo(work_xy.dtype).tiny)
        )
        values = self._appearance[gather_y, gather_x]
        color = (weights[..., None] * values).sum(dim=1)
        color = color / denominator[:, None].clamp_min(torch.finfo(work_xy.dtype).tiny)
        nearest_x = torch.floor(local[:, 0] + 0.5).to(torch.long).clamp(0, width - 1)
        nearest_y = torch.floor(local[:, 1] + 0.5).to(torch.long).clamp(0, height - 1)
        alpha = self._alpha[nearest_y, nearest_x] & valid
        color = torch.where(alpha[:, None], color, torch.zeros_like(color))
        return color.to(target_device), alpha.to(target_device), valid.to(target_device)

    def query_appearance(self, xy: Any) -> tuple[Any, Any, Any]:
        """Query only the continuous appearance/alpha plane without the structural index.

        Placement methods that own an independent geometric likelihood must not pay for, or
        accidentally interpret, the sparse proposal measure as supervision.  This narrow public
        seam exposes the already-tested continuous packet reconstruction while preserving the
        caller's input device for all returned tensors.
        """
        return self._appearance_query(xy)

    def query(self, xy: Any, component_chunk: int = 4096) -> Any:
        color, alpha, valid = self._appearance_query(xy)
        weight_sum = self.structural_backend.query_weight_sum(
            xy, component_chunk=component_chunk
        )
        self._sync_structural_counters()
        weight_sum = torch_where(self._torch, alpha, weight_sum, 0.0)
        numerator = color * weight_sum[:, None]
        return self._observation_query(
            color=color,
            numerator=numerator,
            weight_sum=weight_sum,
            valid=valid,
        )

    def query_weight_sum(self, xy: Any, component_chunk: int = 4096) -> Any:
        _color, alpha, _valid = self._appearance_query(xy)
        weight_sum = self.structural_backend.query_weight_sum(
            xy, component_chunk=component_chunk
        )
        self._sync_structural_counters()
        return torch_where(self._torch, alpha, weight_sum, 0.0)


class CodecNativeAlphaSupportBackend:
    """Placement-only alpha support paired with codec-native appearance queries.

    CORE-016's ordinary backend uses the sparse structural measure as CompactCarve's coverage
    signal.  That is correct for proposal scoring but does not represent a filled silhouette.
    This wrapper keeps the same appearance color and crop validity while replacing only
    ``weight_sum`` with a constant inside the packet's exact alpha mask.  The constant is derived
    from CompactCarve's soft-coverage equation, so ``soft_coverage`` is the value reconstructed by
    ``1 - exp(-area * weight / (field_mass * coverage_scale))`` at every inside query.

    The wrapper deliberately never calls ``structural_backend``.  Sparse structure still chooses
    source rays through ``GaussianObservationField``; alpha supplies placement support; appearance
    supplies radiance.  It is not a faithful standalone observation backend and must not be used as
    a training teacher.
    """

    def __init__(
        self,
        appearance_backend: CodecNativeObservationBackend,
        *,
        coverage_scale: float = 1.0,
        soft_coverage: float = 0.95,
    ) -> None:
        if not isinstance(appearance_backend, CodecNativeObservationBackend):
            raise TypeError("appearance_backend must be CodecNativeObservationBackend")
        if not math.isfinite(coverage_scale) or coverage_scale <= 0.0:
            raise ValueError("coverage_scale must be finite and positive")
        if not math.isfinite(soft_coverage) or not 0.0 < soft_coverage < 1.0:
            raise ValueError("soft_coverage must be finite and lie in (0,1)")

        torch = appearance_backend._torch
        field = appearance_backend.field
        fit_width = int(field.fit_window[2])
        fit_height = int(field.fit_window[3])
        if fit_width <= 0 or fit_height <= 0:
            raise ValueError("the structural field must have a positive fit-window area")
        variances = field.effective_variances()
        mass = (
            field.amplitudes
            * (2.0 * math.pi)
            * variances.prod(dim=1).sqrt()
        ).sum()
        mass_value = float(mass.detach().cpu())
        if not math.isfinite(mass_value) or mass_value <= 0.0:
            raise ValueError("the structural field must have finite positive proposal mass")

        area = float(fit_width * fit_height)
        weight_inside = (
            -math.log1p(-soft_coverage) * coverage_scale * mass_value / area
        )
        if not math.isfinite(weight_inside) or weight_inside <= 0.0:
            raise RuntimeError("derived alpha-support weight is invalid")

        self._torch = torch
        self._observation_query = appearance_backend._observation_query
        self.appearance_backend = appearance_backend
        self.field = field
        self.coverage_scale = float(coverage_scale)
        self.soft_coverage = float(soft_coverage)
        self.field_mass = mass_value
        self.fit_window_area = area
        self.weight_inside = weight_inside
        # This wrapper allocates no index or packet payload of its own.  The referenced appearance
        # tensors remain charged once through the parent packet/backend record.
        self.payload_bytes = 0
        self.reused_payload_bytes = int(getattr(appearance_backend, "payload_bytes", 0))
        self.n_entries = 0
        self.max_candidates = 0
        self.component_id_dtype = None
        self.total_queries = 0
        self.total_points = 0
        self.total_pairs_evaluated = 0
        self.last_pair_chunk = 0
        self.peak_pair_chunk = 0

    def _support_query(self, xy: Any) -> tuple[Any, Any, Any, Any]:
        color, alpha, valid = self.appearance_backend._appearance_query(xy)
        weight = self._torch.where(
            alpha,
            self._torch.full_like(alpha, self.weight_inside, dtype=color.dtype),
            self._torch.zeros_like(alpha, dtype=color.dtype),
        )
        self.total_queries += 1
        self.total_points += int(xy.shape[0])
        return color, weight, alpha, valid

    def query(self, xy: Any, component_chunk: int = 4096) -> Any:
        del component_chunk
        color, weight, _alpha, valid = self._support_query(xy)
        return self._observation_query(
            color=color,
            numerator=color * weight[:, None],
            weight_sum=weight,
            valid=valid,
        )

    def query_weight_sum(self, xy: Any, component_chunk: int = 4096) -> Any:
        del component_chunk
        _color, weight, _alpha, _valid = self._support_query(xy)
        return weight

    def reconstructed_soft_coverage(self) -> float:
        """Return the CompactCarve coverage implied by the derived inside weight."""
        relative_density = self.fit_window_area * self.weight_inside / self.field_mass
        return 1.0 - math.exp(-relative_density / self.coverage_scale)


def torch_where(torch_module: Any, condition: Any, value: Any, other: float) -> Any:
    """Keep the adapter import-lazy while giving mypy/tests one small tensor helper."""
    return torch_module.where(condition, value, torch_module.full_like(value, other))


def make_realtime_gs_view(
    packet: CodecNativeField,
    *,
    device: str = "cpu",
    query_device: str | None = None,
    tile_size: int = 16,
    payload_bytes: int | None = None,
) -> RealtimeGSCodecNativeView:
    """Create the paired current realtime-gs field/backend objects.

    The structural object's component colors are sampled from the authoritative appearance plane
    at their centers, which gives existing lifting initializers meaningful source colors.  Full
    teacher colors still come from ``query_backend``; using ``structural_field.query`` instead is a
    semantic error and is deliberately not hidden by this adapter. ``device`` owns the structural
    metadata while ``query_device`` may independently place the indexed density and appearance
    queries. In particular, ``device="cpu", query_device="cuda"`` matches realtime-gs's CPU
    CompactCarve metadata contract while accelerating its pluggable point queries.
    """
    if not isinstance(packet, CodecNativeField):
        raise TypeError("packet must be a CodecNativeField")
    try:
        import torch
        from rtgs.core.observation2d import GaussianObservationField, GaussianObservationIndex
    except ImportError as exc:
        raise RuntimeError(
            "the codec-native realtime-gs adapter requires torch and the optional realtime-gs "
            "package"
        ) from exc
    structure = packet.structure
    crop = packet.canvas_crop
    means_local = structure.means_xy.astype(np.float32)
    colors = packet.query_appearance(
        means_local, coordinate_space="crop", apply_alpha=False
    ).astype(np.float32)
    target_device = torch.device(device)
    backend_device = target_device if query_device is None else torch.device(query_device)

    def tensor(value: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(value)).to(target_device)

    offset = np.asarray([crop.crop_x + 0.5, crop.crop_y + 0.5], dtype=np.float32)
    native_means = (means_local + offset).astype(np.float32)
    roundtrip_local = (native_means - offset).astype(np.float32)
    mean_residuals = (means_local - roundtrip_local).astype(np.float32)
    field = GaussianObservationField(
        width=crop.canvas_width,
        height=crop.canvas_height,
        means=tensor(native_means),
        log_scales=tensor(structure.log_scales_xy.astype(np.float32)),
        rotations=tensor(structure.rotations_rad.astype(np.float32)),
        colors=tensor(colors),
        amplitudes=tensor(structure.structural_mass.astype(np.float32)),
        mean_residuals=tensor(mean_residuals),
        blend_mode="normalized",
        epsilon=1e-8,
        sigma_cutoff=structure.semantics.support.sigma_cutoff,
        support_fade_alpha=structure.semantics.support.fade_alpha,
        aa_dilation=0.0,
        fit_window=(
            crop.crop_x,
            crop.crop_y,
            crop.crop_width,
            crop.crop_height,
        ),
        n_init=structure.n,
        provider="structsplat",
        producer_version="sgdp-v2",
        producer_source_digest=_producer_source_digest(),
        fit_config_digest=_config_digest(packet),
    )
    if backend_device.type == "cpu":
        if target_device.type != "cpu":
            raise ValueError(
                "a CPU codec-native query backend requires a CPU structural field"
            )
        index = GaussianObservationIndex(field, tile_size=tile_size)
    elif backend_device.type == "cuda":
        from rtgs.core.observation2d_cuda import GaussianObservationIndexCuda

        index = GaussianObservationIndexCuda.from_field(
            field,
            tile_size=tile_size,
            device=backend_device,
        )
    else:
        raise ValueError(
            "the codec-native realtime-gs adapter supports CPU or CUDA query devices"
        )
    backend = CodecNativeObservationBackend(
        packet,
        field,
        index,
        device=backend_device,
        payload_bytes=payload_bytes,
    )
    alpha = torch.from_numpy(packet.alpha_mask.copy()).to(target_device)
    return RealtimeGSCodecNativeView(field, backend, alpha)


def make_alpha_support_backend(
    view: RealtimeGSCodecNativeView,
    *,
    coverage_scale: float = 1.0,
    soft_coverage: float = 0.95,
) -> CodecNativeAlphaSupportBackend:
    """Build CORE-017's no-extra-payload placement-support backend for one paired view."""
    if not isinstance(view, RealtimeGSCodecNativeView):
        raise TypeError("view must be RealtimeGSCodecNativeView")
    if not isinstance(view.query_backend, CodecNativeObservationBackend):
        raise TypeError("view.query_backend must be CodecNativeObservationBackend")
    return CodecNativeAlphaSupportBackend(
        view.query_backend,
        coverage_scale=coverage_scale,
        soft_coverage=soft_coverage,
    )
