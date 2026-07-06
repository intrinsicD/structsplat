"""Feed-forward Gaussian warm-start interface (FF-001).

The learned amortized predictor is intentionally behind this module boundary: callers ask for
`image, InitConfig/options -> GaussianField`, and the implementation may be a checkpoint-backed
model, a saved-field warm start, or a deterministic prior. The first committed slice supplies the
stable interface plus saved-field/tensor-prior predictors so training/export tooling can depend on
one API before learned weights exist.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import numpy as np
import torch

from .config import InitConfig, StructureTensorConfig
from .gaussians import GaussianField
from . import structure_tensor as st


class GaussianPredictor(Protocol):
    """Predict a Gaussian field from an image and initialization options."""

    def predict(
        self,
        img: np.ndarray,
        icfg: InitConfig,
        scfg: StructureTensorConfig | None = None,
        *,
        density: np.ndarray | None = None,
        tensor: st.StructureTensor | None = None,
        device: str = "cpu",
    ) -> GaussianField:
        ...


def _clamp_field_to_image(field: GaussianField, H: int, W: int) -> GaussianField:
    out = field.detached()
    with torch.no_grad():
        out.means[:, 0].clamp_(0.0, max(float(W - 1), 0.0))
        out.means[:, 1].clamp_(0.0, max(float(H - 1), 0.0))
    return out


class TensorPriorPredictor:
    """Deterministic predictor that delegates placement to an existing tensor-prior strategy.

    This is not a learned feed-forward model. It is the fallback implementation that makes the
    FF-001 API executable and gives teacher export/short-refinement code a stable path while
    learned checkpoint support is developed.
    """

    def __init__(self, fallback_strategy: str = "aniso_flanking"):
        if fallback_strategy == "feedforward":
            raise ValueError("fallback_strategy cannot be feedforward")
        self.fallback_strategy = fallback_strategy

    def predict(
        self,
        img: np.ndarray,
        icfg: InitConfig,
        scfg: StructureTensorConfig | None = None,
        *,
        density: np.ndarray | None = None,
        tensor: st.StructureTensor | None = None,
        device: str = "cpu",
    ) -> GaussianField:
        from . import init as _init

        prior_cfg = replace(
            icfg,
            strategy=self.fallback_strategy,
            predictor_checkpoint=None,
            predictor_fallback_strategy=self.fallback_strategy,
        )
        return _init.build_field(img, prior_cfg, scfg, density=density, tensor=tensor,
                                 device=device)


class CheckpointWarmStartPredictor:
    """Warm-start from a saved `GaussianField`, padding with a tensor-prior fallback if needed."""

    def __init__(self, checkpoint: str, fallback_strategy: str = "aniso_flanking"):
        self.checkpoint = checkpoint
        self.fallback = TensorPriorPredictor(fallback_strategy)

    def predict(
        self,
        img: np.ndarray,
        icfg: InitConfig,
        scfg: StructureTensorConfig | None = None,
        *,
        density: np.ndarray | None = None,
        tensor: st.StructureTensor | None = None,
        device: str = "cpu",
    ) -> GaussianField:
        H, W = img.shape[:2]
        field = _clamp_field_to_image(GaussianField.load(self.checkpoint, device=device), H, W)
        budget = int(icfg.num_gaussians)
        if field.n > budget:
            return field.subset(slice(0, budget))
        if field.n == budget:
            return field

        extra_cfg = replace(icfg, num_gaussians=budget - field.n, predictor_checkpoint=None)
        extra = self.fallback.predict(
            img, extra_cfg, scfg, density=density, tensor=tensor, device=device
        )
        return field.append(extra)


def predict_field(
    img: np.ndarray,
    icfg: InitConfig,
    scfg: StructureTensorConfig | None = None,
    *,
    density: np.ndarray | None = None,
    tensor: st.StructureTensor | None = None,
    device: str = "cpu",
) -> GaussianField:
    """Predict a warm-start field for `InitConfig(strategy="feedforward")`.

    Current backends:
    - `predictor_checkpoint`: load a saved `GaussianField` and pad/truncate to the requested N.
    - no checkpoint: deterministic tensor-prior fallback through `predictor_fallback_strategy`.
    """

    if icfg.strategy != "feedforward":
        raise ValueError(f"predict_field expects strategy='feedforward', got {icfg.strategy!r}")
    if icfg.predictor_checkpoint:
        predictor: GaussianPredictor = CheckpointWarmStartPredictor(
            icfg.predictor_checkpoint,
            icfg.predictor_fallback_strategy,
        )
    else:
        predictor = TensorPriorPredictor(icfg.predictor_fallback_strategy)
    return predictor.predict(img, icfg, scfg, density=density, tensor=tensor, device=device)
