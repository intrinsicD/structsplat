"""Feed-forward Gaussian warm-start interface (FF-001).

The learned amortized predictor is intentionally behind this module boundary: callers ask for
`image, InitConfig/options -> GaussianField`, and the implementation may be a checkpoint-backed
model, a saved-field warm start, or a deterministic prior. The first committed slice supplies the
stable interface plus saved-field/tensor-prior predictors so training/export tooling can depend on
one API before learned weights exist.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from .config import InitConfig, StructureTensorConfig
from .gaussians import GaussianField
from . import structure_tensor as st


LEARNED_CHECKPOINT_FORMAT = "structsplat.feedforward.v1"


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


def _logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def image_batch_tensor(
    images: list[np.ndarray] | np.ndarray,
    image_size: int,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Convert one image or a list of HWC float images into resized BCHW tensors."""

    if isinstance(images, np.ndarray):
        batch = [images]
    else:
        batch = images
    tensors = []
    for img in batch:
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 image, got shape {arr.shape}")
        tensors.append(torch.as_tensor(arr, dtype=torch.float32, device=device).permute(2, 0, 1))
    x = torch.stack(tensors, dim=0).clamp(0.0, 1.0)
    return F.interpolate(
        x,
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    )


class TinyGaussianPredictorNet(torch.nn.Module):
    """Compact image-to-Gaussian regressor used by FF-001 smoke training.

    The network is intentionally small: it establishes the learned-checkpoint contract without
    pretending to be the final amortized model. It predicts N independent Gaussian attribute rows
    from a global image encoding.
    """

    def __init__(self, num_gaussians: int, hidden: int = 64):
        super().__init__()
        self.num_gaussians = int(num_gaussians)
        self.hidden = int(hidden)
        if self.num_gaussians <= 0:
            raise ValueError(f"num_gaussians must be positive, got {num_gaussians}")
        if self.hidden < 4:
            raise ValueError(f"hidden must be >= 4, got {hidden}")
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, hidden // 2, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden // 2, hidden, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, self.num_gaussians * 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.encoder(x))
        return raw.reshape(x.shape[0], self.num_gaussians, 10)


def decode_normalized_predictions(
    raw: torch.Tensor,
    *,
    max_scale_norm: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Decode raw network output into normalized Gaussian attributes."""

    if raw.ndim != 3 or raw.shape[-1] != 10:
        raise ValueError(f"expected raw shape BxNx10, got {tuple(raw.shape)}")
    scale_floor = 1e-3
    means = torch.sigmoid(raw[..., 0:2])
    scales = scale_floor + (float(max_scale_norm) - scale_floor) * torch.sigmoid(raw[..., 2:4])
    rot_raw = raw[..., 4:6]
    rot_unit = F.normalize(rot_raw, dim=-1, eps=1e-6)
    colors = torch.sigmoid(raw[..., 6:9])
    opacities = torch.sigmoid(raw[..., 9])
    return {
        "means": means,
        "scales": scales,
        "rot_unit": rot_unit,
        "colors": colors,
        "opacities": opacities,
    }


def field_to_normalized_targets(
    field: GaussianField,
    H: int,
    W: int,
    num_gaussians: int,
    *,
    device: str | torch.device = "cpu",
    max_scale_norm: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Convert a teacher field to normalized targets for feed-forward training."""

    f = field.detached()
    if f.n > num_gaussians:
        f = f.subset(slice(0, num_gaussians))

    def pad(t: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        t = t.to(device=device, dtype=torch.float32)
        if t.shape[0] == num_gaussians:
            return t
        if t.shape[0] > num_gaussians:
            return t[:num_gaussians]
        extra = torch.full(
            (num_gaussians - t.shape[0], *t.shape[1:]),
            value,
            device=device,
            dtype=torch.float32,
        )
        return torch.cat([t, extra], dim=0)

    dims = torch.tensor(
        [max(float(W - 1), 1.0), max(float(H - 1), 1.0)],
        device=device,
        dtype=torch.float32,
    )
    scale_dims = torch.tensor(
        [max(float(W), 1.0), max(float(H), 1.0)],
        device=device,
        dtype=torch.float32,
    )
    means = pad(f.means, 0.0) / dims
    scales = torch.exp(pad(f.log_scales, np.log(1e-3))) / scale_dims
    rotations = pad(f.rotations.reshape(-1, 1), 0.0).reshape(-1)
    colors = pad(f.colors, 0.0).clamp(0.0, 1.0)
    if f.opacities is None:
        opacities = torch.full((f.n,), 0.99, device=device, dtype=torch.float32)
    else:
        opacities = torch.sigmoid(f.opacities.detach().to(device=device, dtype=torch.float32))
    opacities = pad(opacities.reshape(-1, 1), 0.99).reshape(-1)
    return {
        "means": means.clamp(0.0, 1.0),
        "scales": scales.clamp(1e-3, float(max_scale_norm)),
        "rot_unit": torch.stack([torch.cos(rotations), torch.sin(rotations)], dim=1),
        "colors": colors,
        "opacities": opacities.clamp(1e-4, 1.0 - 1e-4),
    }


def normalized_predictions_to_field(
    attrs: dict[str, torch.Tensor],
    H: int,
    W: int,
    *,
    device: str | torch.device = "cpu",
) -> GaussianField:
    """Convert one decoded normalized prediction row into a `GaussianField`."""

    means_norm = attrs["means"]
    if means_norm.ndim == 3:
        means_norm = means_norm[0]
    scales_norm = attrs["scales"]
    if scales_norm.ndim == 3:
        scales_norm = scales_norm[0]
    rot_unit = attrs["rot_unit"]
    if rot_unit.ndim == 3:
        rot_unit = rot_unit[0]
    colors = attrs["colors"]
    if colors.ndim == 3:
        colors = colors[0]
    opacities = attrs["opacities"]
    if opacities.ndim == 2:
        opacities = opacities[0]

    dims = torch.tensor(
        [max(float(W - 1), 1.0), max(float(H - 1), 1.0)],
        device=device,
        dtype=torch.float32,
    )
    scale_dims = torch.tensor(
        [max(float(W), 1.0), max(float(H), 1.0)],
        device=device,
        dtype=torch.float32,
    )
    means = means_norm.to(device=device, dtype=torch.float32) * dims
    scales = (scales_norm.to(device=device, dtype=torch.float32) * scale_dims).clamp_min(1e-3)
    rotations = torch.atan2(
        rot_unit.to(device=device, dtype=torch.float32)[:, 1],
        rot_unit.to(device=device, dtype=torch.float32)[:, 0],
    )
    return GaussianField(
        means,
        torch.log(scales),
        rotations,
        colors.to(device=device, dtype=torch.float32).clamp(0.0, 1.0),
        _logit(opacities.to(device=device, dtype=torch.float32)),
    )


def prediction_loss(
    raw: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    max_scale_norm: float = 0.5,
) -> torch.Tensor:
    """Weighted regression loss for the tiny FF predictor."""

    pred = decode_normalized_predictions(raw, max_scale_norm=max_scale_norm)
    loss = 4.0 * F.mse_loss(pred["means"], targets["means"])
    loss = loss + 2.0 * F.mse_loss(torch.log(pred["scales"]), torch.log(targets["scales"]))
    loss = loss + 0.5 * F.mse_loss(pred["rot_unit"], targets["rot_unit"])
    loss = loss + F.mse_loss(pred["colors"], targets["colors"])
    loss = loss + 0.2 * F.mse_loss(pred["opacities"], targets["opacities"])
    return loss


def make_learned_checkpoint(
    model: TinyGaussianPredictorNet,
    *,
    image_size: int,
    max_scale_norm: float,
    train_config: dict[str, Any] | None = None,
    loss_history: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Build a serializable checkpoint payload for learned feed-forward prediction."""

    safe_train_config = json.loads(json.dumps(train_config or {}, default=str))
    safe_loss_history = json.loads(json.dumps(loss_history or [], default=str))
    return {
        "format": LEARNED_CHECKPOINT_FORMAT,
        "model": "tiny_cnn_gaussian",
        "num_gaussians": int(model.num_gaussians),
        "image_size": int(image_size),
        "hidden": int(model.hidden),
        "max_scale_norm": float(max_scale_norm),
        "state_dict": model.state_dict(),
        "train_config": safe_train_config,
        "loss_history": safe_loss_history,
    }


def load_learned_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("format") != LEARNED_CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported feed-forward checkpoint format {ckpt.get('format')!r}; "
            f"expected {LEARNED_CHECKPOINT_FORMAT!r}"
        )
    return ckpt


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


class LearnedCheckpointPredictor:
    """Warm-start from a tiny learned FF-001 predictor checkpoint."""

    def __init__(self, checkpoint: str, fallback_strategy: str = "aniso_flanking"):
        self.checkpoint = checkpoint
        self.fallback = TensorPriorPredictor(fallback_strategy)

    def _predict_base_field(self, img: np.ndarray, device: str) -> GaussianField:
        H, W = img.shape[:2]
        ckpt = load_learned_checkpoint(self.checkpoint, device=device)
        model = TinyGaussianPredictorNet(
            int(ckpt["num_gaussians"]),
            hidden=int(ckpt.get("hidden", 64)),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            x = image_batch_tensor(img, int(ckpt["image_size"]), device=device)
            raw = model(x)
            attrs = decode_normalized_predictions(
                raw,
                max_scale_norm=float(ckpt.get("max_scale_norm", 0.5)),
            )
            return normalized_predictions_to_field(attrs, H, W, device=device)

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
        field = _clamp_field_to_image(self._predict_base_field(img, device), H, W)
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
    - learned `.pt`/`.pth`: run a tiny trained predictor checkpoint, then pad/truncate to N.
    - `predictor_checkpoint`: load a saved `GaussianField` and pad/truncate to the requested N.
    - no checkpoint: deterministic tensor-prior fallback through `predictor_fallback_strategy`.
    """

    if icfg.strategy != "feedforward":
        raise ValueError(f"predict_field expects strategy='feedforward', got {icfg.strategy!r}")
    if icfg.predictor_checkpoint:
        suffix = Path(icfg.predictor_checkpoint).suffix.lower()
        if suffix in (".pt", ".pth"):
            predictor: GaussianPredictor = LearnedCheckpointPredictor(
                icfg.predictor_checkpoint,
                icfg.predictor_fallback_strategy,
            )
        else:
            predictor = CheckpointWarmStartPredictor(
                icfg.predictor_checkpoint,
                icfg.predictor_fallback_strategy,
            )
    else:
        predictor = TensorPriorPredictor(icfg.predictor_fallback_strategy)
    return predictor.predict(img, icfg, scfg, density=density, tensor=tensor, device=device)
