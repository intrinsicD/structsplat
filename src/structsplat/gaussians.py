"""2D Gaussian field with the rotation-scaling (RS) parameterization (ADR-0002).

Each Gaussian: mean mu in pixel coords, log-scales (log sx, log sy), rotation theta, base color
(3), and optionally local affine color gradients (2,3). RS is preferred over Cholesky because it
decouples orientation from extent, which is exactly what the structure-tensor init wants to set
directly.

Requires torch.
"""
from __future__ import annotations
import numpy as np
import torch


class GaussianField:
    def __init__(self, means, log_scales, rotations, colors, opacities=None, scale_max=None,
                 color_grads=None, background_mask=None, filter_variance=None):
        self.means = means            # (N,2) float
        self.log_scales = log_scales  # (N,2)
        self.rotations = rotations    # (N,)
        self.colors = colors          # base color, (N,3)
        self.opacities = opacities    # optional logits, (N,)
        self.scale_max = scale_max    # optional per-Gaussian optimization cap, (N,2)
        self.color_grads = color_grads  # optional local affine color coeffs, (N,2,3)
        self.background_mask = background_mask  # optional bool mask: frozen-geometry bg rows
        # Optional non-trainable isotropic covariance variance (px^2), one value per row.
        # A generation-density filter is stored separately from the trainable RS scales so each
        # birth cohort keeps the footprint assigned when it was inserted.
        self.filter_variance = filter_variance

    @classmethod
    def from_numpy(cls, means, scales, angles, colors, opacities=None, scale_max=None,
                   device="cpu", dtype=torch.float32, color_grads=None,
                   background_mask=None, filter_variance=None):
        def t(a):
            # Always copy: torch.as_tensor is zero-copy for float32 CPU ndarrays, so without
            # the clone an optimizer step would mutate the caller's init arrays in place
            # (verified via np.shares_memory). The field owns its parameters (CORE-004).
            return torch.as_tensor(np.asarray(a), device=device, dtype=dtype).clone()

        scales = np.clip(np.asarray(scales), 1e-3, None)
        opacity_t = None if opacities is None else t(opacities).reshape(-1)
        scale_max_t = None if scale_max is None else t(scale_max)
        color_grads_t = None if color_grads is None else t(color_grads)
        background_mask_t = None
        if background_mask is not None:
            background_mask_t = torch.as_tensor(
                np.asarray(background_mask), device=device, dtype=torch.bool
            ).reshape(-1).clone()
        filter_variance_t = (
            None if filter_variance is None else t(filter_variance).reshape(-1)
        )
        return cls(t(means), torch.log(t(scales)), t(angles).reshape(-1), t(colors),
                   opacity_t, scale_max_t, color_grads_t, background_mask_t,
                   filter_variance_t)

    @property
    def n(self) -> int:
        return self.means.shape[0]

    @property
    def background_count(self) -> int:
        if self.background_mask is None:
            return 0
        return int(self.background_mask.detach().sum().item())

    def detached(self) -> "GaussianField":
        """Return a leaf-tensor copy suitable for rebuilding an optimizer."""
        return GaussianField(
            self.means.detach().clone(),
            self.log_scales.detach().clone(),
            self.rotations.detach().clone(),
            self.colors.detach().clone(),
            None if self.opacities is None else self.opacities.detach().clone(),
            None if self.scale_max is None else self.scale_max.detach().clone(),
            None if self.color_grads is None else self.color_grads.detach().clone(),
            None if self.background_mask is None else self.background_mask.detach().clone(),
            None if self.filter_variance is None else self.filter_variance.detach().clone(),
        )

    def subset(self, idx) -> "GaussianField":
        return GaussianField(
            self.means.detach()[idx].clone(),
            self.log_scales.detach()[idx].clone(),
            self.rotations.detach()[idx].clone(),
            self.colors.detach()[idx].clone(),
            None if self.opacities is None else self.opacities.detach()[idx].clone(),
            None if self.scale_max is None else self.scale_max.detach()[idx].clone(),
            None if self.color_grads is None else self.color_grads.detach()[idx].clone(),
            None if self.background_mask is None else self.background_mask.detach()[idx].clone(),
            None if self.filter_variance is None else self.filter_variance.detach()[idx].clone(),
        )

    def append(self, other: "GaussianField") -> "GaussianField":
        def cat(a, b):
            return torch.cat([a.detach(), b.detach()], dim=0).clone()

        def cat_optional(a, b):
            if a is None and b is None:
                return None
            # opacities=None renders as opacity 1.0; the missing side must keep that
            # appearance after the append, so pad with a saturating logit (sigmoid(10)
            # ~ 0.99995), not zeros (which would silently mean opacity 0.5)
            if a is None:
                a = torch.full((self.n,), 10.0, device=b.device, dtype=b.dtype)
            if b is None:
                b = torch.full((other.n,), 10.0, device=a.device, dtype=a.dtype)
            return cat(a, b)

        def cat_scale_max(a, b):
            if a is None and b is None:
                return None
            if a is None:
                a = torch.full((self.n, 2), float("inf"), device=b.device, dtype=b.dtype)
            if b is None:
                b = torch.full((other.n, 2), float("inf"), device=a.device, dtype=a.dtype)
            return cat(a, b)

        def cat_color_grads(a, b):
            if a is None and b is None:
                return None
            if a is None:
                a = torch.zeros(self.n, 2, 3, device=b.device, dtype=b.dtype)
            if b is None:
                b = torch.zeros(other.n, 2, 3, device=a.device, dtype=a.dtype)
            return cat(a, b)

        def cat_background_mask(a, b):
            if a is None and b is None:
                return None
            if a is None:
                a = torch.zeros(self.n, device=b.device, dtype=torch.bool)
            if b is None:
                b = torch.zeros(other.n, device=a.device, dtype=torch.bool)
            return torch.cat(
                [a.detach().to(dtype=torch.bool), b.detach().to(device=a.device, dtype=torch.bool)],
                dim=0,
            ).clone()

        def cat_filter_variance(a, b):
            if a is None and b is None:
                return None
            # Missing covariance filtering means exactly zero added variance. This also makes
            # pyramid/appended fields safe: generation-density mode fills zero rows at the next
            # fit entry, while default-off rendering remains unchanged.
            if a is None:
                a = torch.zeros(self.n, device=b.device, dtype=b.dtype)
            if b is None:
                b = torch.zeros(other.n, device=a.device, dtype=a.dtype)
            return cat(a, b)

        return GaussianField(
            cat(self.means, other.means),
            cat(self.log_scales, other.log_scales),
            cat(self.rotations, other.rotations),
            cat(self.colors, other.colors),
            cat_optional(self.opacities, other.opacities),
            cat_scale_max(self.scale_max, other.scale_max),
            cat_color_grads(self.color_grads, other.color_grads),
            cat_background_mask(self.background_mask, other.background_mask),
            cat_filter_variance(self.filter_variance, other.filter_variance),
        )

    def trainable(self) -> "GaussianField":
        params = [self.means, self.log_scales, self.rotations, self.colors]
        if self.opacities is not None:
            params.append(self.opacities)
        if self.color_grads is not None:
            params.append(self.color_grads)
        for p in params:
            p.requires_grad_(True)
        return self

    def parameter_groups(self, lr_means, lr_scales, lr_rot, lr_color, lr_opacity=1e-2):
        groups = [
            {"params": [self.means], "lr": lr_means},
            {"params": [self.log_scales], "lr": lr_scales},
            {"params": [self.rotations], "lr": lr_rot},
            {"params": [self.colors], "lr": lr_color},
        ]
        if self.opacities is not None:
            groups.append({"params": [self.opacities], "lr": lr_opacity})
        if self.color_grads is not None:
            groups.append({"params": [self.color_grads], "lr": lr_color})
        return groups

    def with_affine_colors(self) -> "GaussianField":
        if self.color_grads is not None:
            return self
        return GaussianField(
            self.means,
            self.log_scales,
            self.rotations,
            self.colors,
            self.opacities,
            self.scale_max,
            torch.zeros(self.n, 2, 3, device=self.colors.device, dtype=self.colors.dtype),
            self.background_mask,
            self.filter_variance,
        )

    def scales(self):
        return torch.exp(self.log_scales)

    def effective_scales(self, dilation: float = 0.0):
        """Return RS scales after isotropic covariance filtering and optional AA dilation."""
        if dilation < 0.0:
            raise ValueError(f"dilation must be >= 0, got {dilation}")
        scales = self.scales()
        if self.filter_variance is None and dilation == 0.0:
            return scales
        variance = scales.square() + float(dilation)
        if self.filter_variance is not None:
            variance = variance + self.filter_variance.to(
                device=scales.device, dtype=scales.dtype
            ).reshape(-1, 1)
        return torch.sqrt(variance.clamp_min(1e-12))

    def effective_log_scales(self, dilation: float = 0.0):
        """Differentiable log of :meth:`effective_scales`, used for codec materialization."""
        if self.filter_variance is None and dilation == 0.0:
            return self.log_scales
        return torch.log(self.effective_scales(dilation))

    def materialize_covariance_filter(self) -> "GaussianField":
        """Fold filter variance into RS scales for consumers that cannot encode it separately."""
        if self.filter_variance is None:
            return self
        return GaussianField(
            self.means,
            self.effective_log_scales(),
            self.rotations,
            self.colors,
            self.opacities,
            self.scale_max,
            self.color_grads,
            self.background_mask,
            None,
        )

    def opacity_values(self):
        if self.opacities is None:
            return None
        return torch.sigmoid(self.opacities)

    def conics(self, dilation: float = 0.0):
        """Return (a,b,c): the unique entries of the inverse covariance Sigma^-1=[[a,b],[b,c]].

        `dilation` adds an isotropic EWA-style low-pass to the covariance (Sigma + d*I);
        exact under RS since Sigma + d*I = R diag(sx^2+d, sy^2+d) R^T.
        """
        if dilation < 0.0:
            # A negative dilation can drive sx^2+d <= 0, giving negative inverse variances,
            # exp overflow, and inf/inf = NaN through the normalized division. Fail loudly
            # here rather than silently emitting NaN images mid-fit (CORE-004).
            raise ValueError(f"dilation must be >= 0, got {dilation}")
        s = self.scales()
        sx2 = s[:, 0] ** 2
        sy2 = s[:, 1] ** 2
        if self.filter_variance is not None:
            variance = self.filter_variance.to(device=s.device, dtype=s.dtype).reshape(-1)
            sx2 = sx2 + variance
            sy2 = sy2 + variance
        inv_sx2 = 1.0 / (sx2 + dilation)
        inv_sy2 = 1.0 / (sy2 + dilation)
        c = torch.cos(self.rotations)
        sn = torch.sin(self.rotations)
        a = c * c * inv_sx2 + sn * sn * inv_sy2
        b = c * sn * (inv_sx2 - inv_sy2)
        cc = sn * sn * inv_sx2 + c * c * inv_sy2
        return torch.stack([a, b, cc], dim=1)

    def radii(self, sigma_cutoff: float, dilation: float = 0.0):
        # Per-axis support half-widths (N,2): the AABB of the sigma_cutoff ellipse.
        # Sigma_xx = c^2 sx^2 + s^2 sy^2, Sigma_yy = s^2 sx^2 + c^2 sy^2; an elongated
        # Gaussian gets a tight rectangle instead of a square sized by its major axis.
        # Detached on purpose: radii only set the tile/bbox extent, never the loss gradient.
        if dilation < 0.0:
            raise ValueError(f"dilation must be >= 0, got {dilation}")
        with torch.no_grad():
            s2 = self.scales() ** 2
            if self.filter_variance is not None:
                s2 = s2 + self.filter_variance.to(
                    device=s2.device, dtype=s2.dtype
                ).reshape(-1, 1)
            s2 = s2 + dilation
            c = torch.cos(self.rotations)
            sn = torch.sin(self.rotations)
            var_x = c * c * s2[:, 0] + sn * sn * s2[:, 1]
            var_y = sn * sn * s2[:, 0] + c * c * s2[:, 1]
            r = sigma_cutoff * torch.sqrt(torch.stack([var_x, var_y], dim=1))
            return torch.clamp(r.ceil().long(), min=1)

    @torch.no_grad()
    def save(self, path: str):
        data = {
            "means": self.means.detach().cpu().numpy(),
            "log_scales": self.log_scales.detach().cpu().numpy(),
            "rotations": self.rotations.detach().cpu().numpy(),
            "colors": self.colors.detach().cpu().numpy(),
        }
        if self.opacities is not None:
            data["opacities"] = self.opacities.detach().cpu().numpy()
        if self.scale_max is not None:
            data["scale_max"] = self.scale_max.detach().cpu().numpy()
        if self.color_grads is not None:
            data["color_grads"] = self.color_grads.detach().cpu().numpy()
        if self.background_mask is not None:
            data["background_mask"] = self.background_mask.detach().cpu().numpy()
        if self.filter_variance is not None:
            data["filter_variance"] = self.filter_variance.detach().cpu().numpy()
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str, device="cpu"):
        z = np.load(path)

        def t(k):
            return torch.as_tensor(z[k], device=device, dtype=torch.float32)

        opacities = t("opacities") if "opacities" in z.files else None
        scale_max = t("scale_max") if "scale_max" in z.files else None
        color_grads = t("color_grads") if "color_grads" in z.files else None
        background_mask = (
            torch.as_tensor(z["background_mask"], device=device, dtype=torch.bool)
            if "background_mask" in z.files else None
        )
        filter_variance = t("filter_variance") if "filter_variance" in z.files else None
        return cls(t("means"), t("log_scales"), t("rotations"), t("colors"),
                   opacities, scale_max, color_grads, background_mask, filter_variance)
