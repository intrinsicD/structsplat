"""Text-to-2D-Gaussian generation via latent score distillation (GEN-001).

The public path is:

1. sample one raster image from a frozen text-to-image diffusion pipeline;
2. fit that image with the existing StructSplat init/fitter;
3. refine the resulting GaussianField with SDS through the differentiable renderer.

Diffusers is imported lazily so the core package and tests do not require the optional [gen] extra.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
import time
from typing import Any

import numpy as np
import torch

from . import init as _init
from .config import FitConfig, GenConfig, InitConfig, StructureTensorConfig
from .fit import fit
from .gaussians import GaussianField
from .render import render_field


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unknown dtype {name!r}; expected float32, float16, or bfloat16")


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        gen = torch.Generator(device=device)
    except (TypeError, RuntimeError):
        gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def _save_image(path: str, arr: np.ndarray | torch.Tensor) -> None:
    from PIL import Image

    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    a = np.asarray(arr, dtype=np.float32)
    a = np.rint(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(a).save(path)


def _coerce_image(image: Any, cfg: GenConfig) -> np.ndarray:
    if isinstance(image, str):
        from PIL import Image

        pil = Image.open(image).convert("RGB")
        if pil.size != (cfg.width, cfg.height):
            pil = pil.resize((cfg.width, cfg.height), Image.Resampling.LANCZOS)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
    elif torch.is_tensor(image):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] != 3:
        raise ValueError(f"expected RGB image, got shape {arr.shape}")
    arr = arr.astype(np.float32)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    if arr.shape[:2] != (cfg.height, cfg.width):
        from PIL import Image

        pil = Image.fromarray(np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))
        pil = pil.resize((cfg.width, cfg.height), Image.Resampling.LANCZOS)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def render_gaussian_field(
    field: GaussianField,
    cfg: GenConfig,
    height: int | None = None,
    width: int | None = None,
    clamp: bool = False,
) -> torch.Tensor:
    """Render a field with the GEN-001 renderer settings."""
    H = int(cfg.height if height is None else height)
    W = int(cfg.width if width is None else width)
    img = render_field(
        field.means,
        field.conics(cfg.aa_dilation),
        field.colors,
        field.radii(cfg.sigma_cutoff, cfg.aa_dilation),
        H,
        W,
        cfg.render_chunk,
        cfg.renderer,
        field.opacity_values(),
        scales=field.effective_scales(
            cfg.aa_dilation if cfg.renderer in ("gsplat", "cuda_gsplat") else 0.0
        ),
        rotations=field.rotations,
        support_fade=cfg.support_fade,
        sigma_cutoff=cfg.sigma_cutoff,
    )
    return img.clamp(0.0, 1.0) if clamp else img


def _bchw_unit(img_hwc: torch.Tensor) -> torch.Tensor:
    return img_hwc.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def rescale_field(
    field: GaussianField,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> GaussianField:
    """Return a detached field whose pixel-coordinate geometry is scaled to a new canvas."""
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    dev, dt = field.means.device, field.means.dtype
    scale = torch.tensor([sx, sy], device=dev, dtype=dt)
    means = field.means.detach().clone() * scale

    # Pixel-coordinate resizing transforms covariance as D Sigma D, not by independently
    # multiplying the Gaussian's *local* RS axes.  Those operations agree only for uniform
    # scaling or axis-aligned Gaussians.  Re-diagonalize the transformed 2x2 covariance so a
    # rotated anisotropic (and covariance-filtered) field remains exact after a nonuniform resize.
    # CPU/CUDA linalg kernels do not consistently support float16/bfloat16 eigendecomposition,
    # while generation defaults to float16.  Do covariance algebra in float32 (float64 stays
    # float64) and cast the resulting RS parameters back to the field dtype.
    work_dtype = torch.float64 if dt == torch.float64 else torch.float32
    # Build the variance after upcasting. ``effective_scales`` squares the base scale in
    # the field dtype, so a perfectly representable float16 scale above sqrt(65504) would
    # overflow before the later float32 conversion.
    log_scales_work = field.log_scales.detach().to(work_dtype)
    effective_variance = torch.exp(2.0 * log_scales_work)
    if field.filter_variance is not None:
        effective_variance = effective_variance + field.filter_variance.detach().to(
            work_dtype
        ).reshape(-1, 1)
    if math.isclose(sx, sy, rel_tol=1e-12, abs_tol=1e-12):
        log_scales = (
            0.5 * torch.log(effective_variance.clamp_min(1e-12)) + math.log(sx)
        ).to(dt)
        rotations = field.rotations.detach().clone()
        scale_max = (
            None if field.scale_max is None
            else field.scale_max.detach().clone() * sx
        )
    else:
        rotations_work = field.rotations.detach().to(work_dtype)
        c = torch.cos(rotations_work)
        s = torch.sin(rotations_work)
        var_x = c.square() * effective_variance[:, 0] + s.square() * effective_variance[:, 1]
        var_y = s.square() * effective_variance[:, 0] + c.square() * effective_variance[:, 1]
        cov_xy = c * s * (effective_variance[:, 0] - effective_variance[:, 1])
        covariance = torch.stack(
            [
                torch.stack([var_x * (sx * sx), cov_xy * (sx * sy)], dim=1),
                torch.stack([cov_xy * (sx * sy), var_y * (sy * sy)], dim=1),
            ],
            dim=1,
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        # eigh is ascending.  Store the major eigenpair as the first RS axis; eigenvector sign
        # is immaterial because Gaussian rotations are pi-periodic.
        eigenvalues = eigenvalues.flip(dims=(1,)).clamp_min(1e-12)
        major = eigenvectors[:, :, 1]
        log_scales = (0.5 * torch.log(eigenvalues)).to(dt)
        rotations = torch.atan2(major[:, 1], major[:, 0]).to(dt)
        if field.scale_max is None:
            scale_max = None
        else:
            # A per-local-axis optimizer cap has no exact representation after D rotates the
            # covariance eigensystem.  Preserve a conservative isotropic upper bound instead of
            # silently applying the old axes to the new rotation.
            cap = field.scale_max.detach().amax(dim=1) * max(sx, sy)
            scale_max = cap[:, None].expand(-1, 2).clone()
    return GaussianField(
        means,
        log_scales,
        rotations,
        field.colors.detach().clone(),
        None if field.opacities is None else field.opacities.detach().clone(),
        scale_max,
        None if field.color_grads is None else field.color_grads.detach().clone(),
        None if field.background_mask is None else field.background_mask.detach().clone(),
        None,
    )


def _long_side_size(src_h: int, src_w: int, long_side: int) -> tuple[int, int]:
    scale = float(long_side) / float(max(src_h, src_w))
    return max(1, int(round(src_h * scale))), max(1, int(round(src_w * scale)))


def _field_params(field: GaussianField) -> list[torch.Tensor]:
    params = [field.means, field.log_scales, field.rotations, field.colors]
    if field.opacities is not None:
        params.append(field.opacities)
    return [p for p in params if p.requires_grad]


def _grad_norm(p: torch.Tensor | None) -> float:
    if p is None or p.grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(p.grad.detach()).cpu())


def _grad_stats(field: GaussianField) -> dict[str, float]:
    return {
        "grad_means": _grad_norm(field.means),
        "grad_scales": _grad_norm(field.log_scales),
        "grad_rot": _grad_norm(field.rotations),
        "grad_color": _grad_norm(field.colors),
        "grad_opacity": _grad_norm(field.opacities),
    }


def _regularization(field: GaussianField, cfg: GenConfig) -> torch.Tensor:
    loss = field.means.new_tensor(0.0)
    if cfg.opacity_l1 > 0.0 and field.opacities is not None:
        loss = loss + cfg.opacity_l1 * field.opacity_values().mean()
    if cfg.scale_l2 > 0.0:
        loss = loss + cfg.scale_l2 * (field.log_scales ** 2).mean()
    if cfg.color_l2 > 0.0:
        loss = loss + cfg.color_l2 * (field.colors ** 2).mean()
    return loss


@torch.no_grad()
def _clamp_field(field: GaussianField, cfg: GenConfig) -> None:
    field.means[:, 0].clamp_(0.0, float(cfg.width - 1))
    field.means[:, 1].clamp_(0.0, float(cfg.height - 1))
    max_scale = (
        float(cfg.max_scale)
        if cfg.max_scale is not None
        else float(max(cfg.height, cfg.width))
    )
    scales = torch.exp(field.log_scales)
    scales = torch.clamp(scales, min=float(cfg.min_scale), max=max_scale)
    if field.scale_max is not None:
        scales = torch.minimum(scales, torch.clamp(field.scale_max, min=float(cfg.min_scale)))
    field.log_scales.copy_(torch.log(torch.clamp(scales, min=float(cfg.min_scale))))
    field.colors.clamp_(float(cfg.color_min), float(cfg.color_max))
    if field.opacities is not None:
        field.opacities.clamp_(-12.0, 12.0)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class StableDiffusionLatentSDS:
    """Thin latent-SDS adapter around a Diffusers Stable Diffusion pipeline."""

    def __init__(self, pipe: Any, noise_scheduler: Any, cfg: GenConfig, device: torch.device):
        self.pipe = pipe
        self.noise_scheduler = noise_scheduler
        self.cfg = cfg
        self.device = device
        self.generator = _make_generator(device, cfg.seed + 17)
        self._prompt_cache: dict[tuple[str, str, bool], torch.Tensor] = {}
        self._check_components()

    @classmethod
    def from_config(cls, cfg: GenConfig) -> "StableDiffusionLatentSDS":
        try:
            from diffusers import DDPMScheduler, DiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "GEN-001 requires the optional generation dependencies. "
                "Install them with `pip install -e '.[gen]'`."
            ) from exc

        device = _resolve_device(cfg.device)
        dtype = torch.float32 if device.type == "cpu" else _torch_dtype(cfg.dtype)
        pipe = DiffusionPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.to(device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        try:
            noise_scheduler = DDPMScheduler.from_pretrained(cfg.model_id, subfolder="scheduler")
        except Exception:
            noise_scheduler = pipe.scheduler
        if hasattr(noise_scheduler, "alphas_cumprod"):
            noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

        for name in ("vae", "unet", "text_encoder"):
            module = getattr(pipe, name, None)
            if module is None:
                continue
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)
        return cls(pipe, noise_scheduler, cfg, device)

    def _check_components(self) -> None:
        missing = [
            name for name in ("vae", "unet", "tokenizer", "text_encoder")
            if getattr(self.pipe, name, None) is None
        ]
        if missing:
            raise RuntimeError(
                "GEN-001 latent SDS currently expects a Stable Diffusion-class pipeline; "
                f"missing components: {', '.join(missing)}"
            )
        if not hasattr(self.noise_scheduler, "add_noise"):
            raise RuntimeError("the selected noise scheduler does not expose add_noise()")

    def sample_raster(self, prompt: str, cfg: GenConfig | None = None) -> np.ndarray:
        cfg = cfg or self.cfg
        gen = _make_generator(self.device, cfg.seed)
        with torch.inference_mode():
            out = self.pipe(
                prompt,
                negative_prompt=cfg.negative_prompt or None,
                height=cfg.height,
                width=cfg.width,
                num_inference_steps=cfg.sample_steps,
                guidance_scale=cfg.sample_guidance_scale,
                generator=gen,
                output_type="np",
            )
        img = out.images[0]
        return _coerce_image(img, cfg)

    def _prompt_embeds(self, prompt: str, do_cfg: bool) -> torch.Tensor:
        key = (prompt, self.cfg.negative_prompt, do_cfg)
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached

        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        def encode(text: str) -> torch.Tensor:
            tokens = tokenizer(
                [text],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(self.device)
            attention_mask = None
            if getattr(getattr(text_encoder, "config", None), "use_attention_mask", False):
                attention_mask = tokens.attention_mask.to(self.device)
            with torch.no_grad():
                out = text_encoder(input_ids, attention_mask=attention_mask)
            return out[0] if isinstance(out, tuple) else out.last_hidden_state

        prompt_embeds = encode(prompt)
        if do_cfg:
            negative = encode(self.cfg.negative_prompt or "")
            prompt_embeds = torch.cat([negative, prompt_embeds], dim=0)
        self._prompt_cache[key] = prompt_embeds
        return prompt_embeds

    def _encode_image_latents(self, image_bchw: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        dtype = next(vae.parameters()).dtype
        x = image_bchw.to(device=self.device, dtype=dtype).clamp(0.0, 1.0) * 2.0 - 1.0
        encoded = vae.encode(x)
        latent_dist = getattr(encoded, "latent_dist", None)
        if latent_dist is None and isinstance(encoded, tuple):
            latent_dist = encoded[0]
        if latent_dist is None:
            raise RuntimeError("VAE encode() did not return a latent distribution")
        latents = latent_dist.mode() if hasattr(latent_dist, "mode") else latent_dist.sample()
        scaling = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))
        return latents * scaling

    def _alphas(self, timesteps: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        alphas = getattr(self.noise_scheduler, "alphas_cumprod", None)
        if alphas is None:
            raise RuntimeError("noise scheduler must expose alphas_cumprod for SDS weighting")
        alphas = alphas.to(device=timesteps.device, dtype=dtype)
        return alphas[timesteps.clamp(0, alphas.numel() - 1)]

    def _prediction_to_epsilon(
        self,
        model_pred: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        prediction_type = _config_get(
            getattr(self.noise_scheduler, "config", {}), "prediction_type", "epsilon"
        )
        if prediction_type == "epsilon":
            return model_pred
        alphas = self._alphas(timesteps, model_pred.dtype).view(-1, 1, 1, 1)
        sigma = torch.sqrt(torch.clamp(1.0 - alphas, min=1e-8))
        alpha = torch.sqrt(torch.clamp(alphas, min=1e-8))
        if prediction_type == "v_prediction":
            return alpha * model_pred + sigma * noisy_latents
        if prediction_type == "sample":
            return (noisy_latents - alpha * model_pred) / sigma
        raise ValueError(f"unsupported scheduler prediction_type {prediction_type!r}")

    def sds_loss(
        self,
        image_bchw: torch.Tensor,
        prompt: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.cfg
        latents = self._encode_image_latents(image_bchw)
        batch = latents.shape[0]
        num_train = int(
            _config_get(
                getattr(self.noise_scheduler, "config", {}),
                "num_train_timesteps",
                1000,
            )
        )
        t_min = min(max(0, cfg.sds_t_min), num_train - 2)
        t_max = min(max(t_min + 1, cfg.sds_t_max), num_train - 1)
        timesteps = torch.randint(
            t_min,
            t_max + 1,
            (batch,),
            device=self.device,
            dtype=torch.long,
            generator=self.generator,
        )
        noise = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=self.generator,
        )
        noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)

        do_cfg = cfg.sds_guidance_scale > 1.0
        model_noisy = torch.cat([noisy, noisy], dim=0) if do_cfg else noisy
        model_in = model_noisy
        model_t = timesteps.repeat(2) if do_cfg else timesteps
        if hasattr(self.noise_scheduler, "scale_model_input"):
            model_in = self.noise_scheduler.scale_model_input(model_in, model_t)
        embeds = self._prompt_embeds(prompt, do_cfg).to(device=self.device, dtype=model_in.dtype)
        with torch.no_grad():
            noise_pred = self.pipe.unet(model_in, model_t, encoder_hidden_states=embeds).sample
        eps_pred = self._prediction_to_epsilon(noise_pred, model_noisy, model_t)
        if do_cfg:
            eps_uncond, eps_text = eps_pred.chunk(2, dim=0)
            eps_pred = eps_uncond + cfg.sds_guidance_scale * (eps_text - eps_uncond)

        alphas = self._alphas(timesteps, latents.dtype).view(batch, 1, 1, 1)
        grad = cfg.sds_weight * (1.0 - alphas) * (eps_pred - noise)
        grad = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        loss = (latents.float() * grad.float()).mean()
        stats = {
            "timestep_mean": float(timesteps.float().mean().detach().cpu()),
            "sds_grad_norm": float(torch.linalg.vector_norm(grad.float()).detach().cpu()),
            "latent_norm": float(torch.linalg.vector_norm(latents.float()).detach().cpu()),
        }
        return loss, stats


def refine_field(
    field: GaussianField,
    guidance: Any,
    cfg: GenConfig,
    prompt: str,
    verbose: bool = True,
) -> dict[str, Any]:
    """Refine an initialized field with a guidance object exposing sds_loss(image_bchw, prompt)."""
    field.trainable()
    opt = torch.optim.Adam(
        field.parameter_groups(
            cfg.lr_means,
            cfg.lr_scales,
            cfg.lr_rot,
            cfg.lr_color,
            cfg.lr_opacity,
        )
    )
    hist: dict[str, list[float]] = {
        "iter": [],
        "loss": [],
        "sds_loss": [],
        "reg_loss": [],
        "elapsed": [],
        "grad_means": [],
        "grad_scales": [],
        "grad_rot": [],
        "grad_color": [],
        "grad_opacity": [],
        "sds_grad_norm": [],
        "timestep_mean": [],
    }
    start = time.time()
    log_every = max(1, cfg.refine_steps // 10) if cfg.refine_steps > 0 else 1
    last_stats: dict[str, float] = {}

    for it in range(cfg.refine_steps):
        opt.zero_grad(set_to_none=True)
        render = render_gaussian_field(field, cfg, clamp=True)
        sds_loss, stats = guidance.sds_loss(_bchw_unit(render), prompt)
        reg_loss = _regularization(field, cfg)
        loss = sds_loss + reg_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite generation loss at iter {it}: {float(loss)}")
        loss.backward()
        grad_stats = _grad_stats(field)
        for p in _field_params(field):
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"non-finite generation gradient at iter {it}")
        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(_field_params(field), cfg.grad_clip_norm)
        opt.step()
        _clamp_field(field, cfg)

        last_stats = {k: float(v) for k, v in stats.items()}
        if it == 0 or (it + 1) % log_every == 0 or it + 1 == cfg.refine_steps:
            hist["iter"].append(float(it + 1))
            hist["loss"].append(float(loss.detach().cpu()))
            hist["sds_loss"].append(float(sds_loss.detach().cpu()))
            hist["reg_loss"].append(float(reg_loss.detach().cpu()))
            hist["elapsed"].append(float(time.time() - start))
            for key, value in grad_stats.items():
                hist[key].append(float(value))
            hist["sds_grad_norm"].append(float(stats.get("sds_grad_norm", 0.0)))
            hist["timestep_mean"].append(float(stats.get("timestep_mean", 0.0)))
            if verbose:
                print(
                    f"gen {it + 1:04d}/{cfg.refine_steps} "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"grad={float(stats.get('sds_grad_norm', 0.0)):.3f}"
                )

    final_render = render_gaussian_field(field, cfg, clamp=True)
    return {
        "field": field,
        "render": final_render,
        "history": hist,
        "stats": last_stats,
    }


def fit_raster_sample(
    sample: np.ndarray,
    cfg: GenConfig,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    device = _resolve_device(str(device) if device is not None else cfg.device)
    icfg = InitConfig(
        strategy=cfg.init_strategy,
        num_gaussians=cfg.num_gaussians,
        opacity_mode="constant",
        init_opacity=cfg.init_opacity,
        seed=cfg.seed,
    )
    scfg = StructureTensorConfig()
    field = _init.build_field(sample, icfg, scfg, device=str(device))
    if cfg.fit_iters <= 0:
        render = render_gaussian_field(field, cfg, clamp=True)
        return {"field": field, "render": render, "fit_seconds": 0.0}

    target = torch.as_tensor(sample, device=device, dtype=torch.float32)
    fcfg = FitConfig(
        iters=cfg.fit_iters,
        lr_means=cfg.lr_means,
        lr_scales=cfg.lr_scales,
        lr_rot=cfg.lr_rot,
        lr_color=cfg.lr_color,
        lr_opacity=cfg.lr_opacity,
        renderer=cfg.renderer,
        render_chunk=cfg.render_chunk,
        sigma_cutoff=cfg.sigma_cutoff,
        support_fade=cfg.support_fade,
        aa_dilation=cfg.aa_dilation,
        log_every=max(1, cfg.fit_iters // 10),
    )
    return fit(field, target, fcfg, verbose=verbose)


def save_generation_result(
    result: dict[str, Any],
    outdir: str,
    prompt: str,
    cfg: GenConfig,
) -> dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    paths: dict[str, str] = {}

    def path(name: str) -> str:
        p = os.path.join(outdir, name)
        paths[name] = p
        return p

    _save_image(path("sample.png"), result["sample"])
    _save_image(path("fit.png"), result["fit_render"])
    _save_image(path("refined.png"), result["render"])
    field_path = path("field.npz")
    result["field"].save(field_path)

    for long_side in cfg.save_resolutions:
        if long_side <= 0:
            continue
        h, w = _long_side_size(cfg.height, cfg.width, int(long_side))
        scaled = rescale_field(result["field"], cfg.height, cfg.width, h, w)
        img = render_gaussian_field(scaled, cfg, height=h, width=w, clamp=True)
        _save_image(path(f"render_{int(long_side)}px.png"), img)

    metadata = {
        "prompt": prompt,
        "config": asdict(cfg),
        "history": result.get("history", {}),
        "stats": result.get("stats", {}),
    }
    with open(path("metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    result["paths"] = paths
    return paths


def generate(
    prompt: str,
    cfg: GenConfig | None = None,
    guidance: Any | None = None,
    sample_image: Any | None = None,
    outdir: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the raster-sample -> fit -> latent-SDS-refine pipeline."""
    cfg = cfg or GenConfig()
    device = _resolve_device(cfg.device)
    if guidance is None:
        guidance = StableDiffusionLatentSDS.from_config(cfg)
    if sample_image is None:
        if not hasattr(guidance, "sample_raster"):
            raise ValueError("guidance must expose sample_raster() when sample_image is not given")
        sample = guidance.sample_raster(prompt, cfg)
    else:
        sample = _coerce_image(sample_image, cfg)

    fit_out = fit_raster_sample(sample, cfg, device=device, verbose=verbose)
    field = fit_out["field"]
    fit_render = fit_out["render"].detach().clone()
    if cfg.refine_steps > 0:
        refine_out = refine_field(field, guidance, cfg, prompt, verbose=verbose)
    else:
        refine_out = {
            "field": field,
            "render": render_gaussian_field(field, cfg, clamp=True),
            "history": {},
            "stats": {},
        }

    result = {
        "prompt": prompt,
        "sample": sample,
        "fit": fit_out,
        "fit_render": fit_render,
        "field": refine_out["field"],
        "render": refine_out["render"],
        "history": refine_out.get("history", {}),
        "stats": refine_out.get("stats", {}),
    }
    if outdir is not None:
        save_generation_result(result, outdir, prompt, cfg)
    return result


__all__ = [
    "StableDiffusionLatentSDS",
    "fit_raster_sample",
    "generate",
    "refine_field",
    "render_gaussian_field",
    "rescale_field",
    "save_generation_result",
]
