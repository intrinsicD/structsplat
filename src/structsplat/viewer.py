"""Bridge to the external igsv browser viewer (interactive-gs-viewer repo).

Embeds the 2D image-plane Gaussian field in 3D so the WebGPU splat viewer can
display it, live during ``fit()`` or as a one-off (ADR-0018):

- positions: ``(x, y)`` pixel coords -> ``(x, (H-1) - y, 0)`` so the image reads
  y-up in the viewer instead of mirrored;
- orientation: the y-flip negates handedness, so ``theta`` maps to a rotation of
  ``-theta`` about +Z;
- scales: ``(sx, sy)`` from the RS parameterization plus the generation-density
  ``filter_variance`` folded in in quadrature; a hair of z-thickness keeps the
  EWA projection finite from edge-on views;
- opacity: sigmoid of the optional logits (the renderer's per-Gaussian weight),
  1.0 when the field has no opacities;
- color: clamped base colors as SH DC coefficients.

The viewer alpha-composites (OVER) while StructSplat's renderer is a normalized
weighted sum (ADR-0003), and local affine color carriers (``color_grads``) are
not reproduced — the browser view is a *diagnostic*, never evidence. igsv is an
optional dependency, imported lazily; importing this module requires torch only.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .gaussians import GaussianField

if TYPE_CHECKING:  # pragma: no cover - typing only
    import igsv

SH_C0 = 0.28209479177387814
Z_THICKNESS_PX = 1e-2


def _require_igsv() -> Any:
    try:
        import igsv
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "live viewing requires the optional igsv package from the "
            "interactive-gs-viewer repository: pip install -e <path>/interactive-gs-viewer/server"
        ) from exc
    return igsv


def field_to_frame(field: GaussianField, image_hw: tuple[int, int],
                   name: str = "field") -> "igsv.SplatFrame":
    """Convert a GaussianField to an igsv SplatFrame (detached CPU copy)."""
    igsv_mod = _require_igsv()
    h, _w = image_hw
    with torch.no_grad():
        means = field.means.detach().cpu().numpy()
        theta = field.rotations.detach().cpu().numpy().astype(np.float64)
        scales_xy = field.log_scales.detach().exp().cpu().numpy().astype(np.float64)
        if field.filter_variance is not None:
            var = field.filter_variance.detach().cpu().numpy().astype(np.float64)
            scales_xy = np.sqrt(scales_xy**2 + var[:, None])
        if field.opacities is not None:
            opacities = torch.sigmoid(field.opacities.detach()).cpu().numpy()
        else:
            opacities = np.ones(field.n, dtype=np.float32)
        colors = field.colors.detach().cpu().numpy()

    n = field.n
    positions = np.zeros((n, 3), dtype=np.float32)
    positions[:, 0] = means[:, 0]
    positions[:, 1] = (h - 1.0) - means[:, 1]

    half = -theta * 0.5  # y-flip negates the in-plane angle; rotation about +Z
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = np.cos(half)
    quats[:, 3] = np.sin(half)

    scales = np.full((n, 3), Z_THICKNESS_PX, dtype=np.float32)
    scales[:, :2] = scales_xy

    sh0 = (np.clip(colors, 0.0, 1.0) - 0.5) / SH_C0

    return igsv_mod.SplatFrame(
        positions=positions,
        quats_wxyz=quats,
        scales=scales,
        opacities=opacities.astype(np.float32),
        sh0=sh0.astype(np.float32),
        name=name,
    )


class LiveFitViewer:
    """igsv server wrapper whose :meth:`observer` plugs into ``fit()``.

    Usage::

        viewer = LiveFitViewer((H, W), port=8890)
        viewer.start()
        fit(field, target, cfg, iteration_observer=viewer.observer, observer_every=25)
        viewer.stop()

    Publishing is coalescing on the server side: a slow browser drops
    intermediate snapshots instead of stalling the fit.
    """

    def __init__(self, image_hw: tuple[int, int], host: str = "127.0.0.1",
                 port: int = 8890, *, token: str | None = None,
                 name: str = "structsplat") -> None:
        igsv_mod = _require_igsv()
        self._igsv = igsv_mod
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.name = name
        self._server = igsv_mod.ViewerServer(host=host, port=port, token=token)
        self._t0 = time.perf_counter()
        self._last: tuple[int, float] | None = None

    @property
    def server(self) -> Any:
        """The underlying igsv ViewerServer."""
        return self._server

    @property
    def url(self) -> str:
        return self._server.url

    def start(self) -> None:
        self._server.start_background()

    def stop(self) -> None:
        self._server.stop()

    def publish(self, field: GaussianField, iteration: int = 0, *,
                loss: float = float("nan"), psnr: float = float("nan")) -> None:
        now = time.perf_counter()
        iters_per_sec = float("nan")
        if self._last is not None and now > self._last[1] and iteration > self._last[0]:
            iters_per_sec = (iteration - self._last[0]) / (now - self._last[1])
        self._last = (iteration, now)
        frame = field_to_frame(field, self.image_hw, name=self.name)
        stats = self._igsv.TrainStats(
            loss=loss, iters_per_sec=iters_per_sec, psnr=psnr,
            wall_seconds=now - self._t0,
        )
        self._server.publish(frame, step=iteration, stats=stats)

    def observer(self, field: GaussianField, iteration: int, loss: float) -> None:
        """Signature matches ``fit(iteration_observer=...)``."""
        self.publish(field, iteration, loss=loss)

    def __enter__(self) -> "LiveFitViewer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
