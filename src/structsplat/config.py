"""Typed configuration objects. Pure-Python (no torch), safe to import anywhere."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class StructureTensorConfig:
    grad_sigma: float = 1.0          # pre-smoothing of the image before gradients
    tensor_sigma: float = 2.0        # rho: smoothing of the outer-product tensor field
    gradient_operator: str = "central"  # central, sobel, or scharr
    color_space: str = "luma"        # luma (Rec.709 gray) or rgb (Di Zenzo multi-channel sum)
    # classification thresholds, expressed as fractions of the 99th-percentile energy
    flat_frac: float = 0.02          # energy below this -> flat
    corner_frac: float = 0.15        # smaller eigenvalue above this*ref -> corner


@dataclass
class InitConfig:
    strategy: str = "aniso_flanking"  # see structsplat.init.STRATEGIES
    num_gaussians: int = 20000
    candidate_oversample: float = 6.0  # WSE draws oversample*N candidates
    density_base: float = 0.05         # floor so flat regions still get some coverage
    density_power: float = 1.0         # density ~ energy**power
    density_mode: str = "structure"    # structure, gradient, variance, hybrid, or uniform
    # wse, density_random, floyd_steinberg, jittered_grid, dart_throwing, halton,
    # farthest_point, or cvt
    sampling_mode: str = "wse"
    # anisotropy: axis ratio cap for edge Gaussians (major/minor); 1.0 => isotropic
    max_axis_ratio: float = 6.0
    coherence_power: float = 1.0     # maps coherence -> anisotropy; >1 is more conservative
    orientation_mode: str = "tensor"   # tensor (strategy default), random, or zero
    scale_mode: str = "spacing"        # spacing, uniform, or knn
    init_scale_mult: float = 1.0       # multiply local spacing to get initial std
    scale_cap_mode: str = "none"       # none, hard, or feature
    scale_cap_max: float | None = None # absolute sigma cap, in pixels
    scale_feature_sigma: float = 3.0   # feature mode: visible half-length ~= sigma*value
    scale_feature_min: float = 0.75    # minimum adaptive sigma cap
    scale_feature_energy_frac: float = 0.25  # stop edge run when energy drops below this local frac
    flank_offset_frac: float = 0.5     # edge center offset in units of local minor spacing
    color_mode: str = "bilinear"       # bilinear, local_mean, two_sided, or aggregate (quadtree)
    color_radius: float = 1.5          # local mean radius or extra side-sample offset
    opacity_mode: str = "none"         # none or constant
    init_opacity: float = 0.9
    seed: int = 0

    def __post_init__(self):
        # WSE draws candidate_oversample * N candidates then reduces to N; < 1 cannot supply N
        # candidates and silently breaks the exact-N contract (INIT-005).
        if self.candidate_oversample < 1.0:
            raise ValueError(
                f"candidate_oversample must be >= 1, got {self.candidate_oversample}")


@dataclass
class FitConfig:
    iters: int = 2000
    # LRs are ~pixels/step for means under Adam; the old 2e-3 left positions nearly frozen.
    # Retuned on the init sweep (ADR-0008): ~6x faster to a target PSNR, +2 dB at fixed iters.
    lr_means: float = 5e-2
    lr_scales: float = 3e-2
    lr_rot: float = 1e-2
    lr_color: float = 3e-2
    lr_opacity: float = 1e-2
    optimizer: str = "adam"            # adam, adamw, or adan
    pixel_loss: str = "l1"             # l1, l2, or charbonnier; SSIM term is mixed separately
    charbonnier_eps: float = 1e-3
    loss_warmup_iters: int = 0
    loss_warmup_pixel_loss: str = "l2"
    ssim_weight: float = 0.3           # loss = (1-w)*L1 + w*(1-SSIM); Instant-GI/AIR default
    ssim_backend: str = "builtin"      # builtin, fused (optional), or auto
    compute_lpips: bool = False        # opt-in: loads a separate AlexNet LPIPS model
    target_psnr: float | None = None   # record iters-to-target if set
    target_psnrs: list[float] = field(default_factory=list)
    log_every: int = 100
    sigma_cutoff: float = 3.0          # render support radius in std devs
    support_fade: bool = False         # C0 compact support: subtract Gaussian tail at cutoff
    aa_dilation: float = 0.0           # EWA-style low-pass: render with Sigma + d*I (px^2)
    render_chunk: int = 512            # reference renderer: max(render_chunk,64)*4096 elements
    renderer: str = "normalized"       # normalized/additive/cuda/cuda_tiled/gsplat variants
    color_basis: str = "constant"      # constant or affine local color model
    color_grad_l2: float = 1e-4        # affine coefficient L2 regularization
    color_solve_every: int | None = None  # periodic fixed-geometry RGB least-squares solve
    color_solve_lambda: float = 1e-4       # Tikhonov pull toward pre-solve colors
    color_solve_maxiter: int = 32          # CG iterations for the color normal equations
    lr_schedule: str = "none"          # none, step, or cosine
    lr_decay_every: int | None = None
    lr_decay_gamma: float = 0.5
    prune_every: int | None = None
    # unnormalized weight sum; <=0 disables pruning. When opacities are present the criterion is
    # opacity-weighted (activity * sigmoid(opacity)), so the threshold is in the same weight-sum
    # units scaled by opacity — a fully transparent Gaussian scores 0 and is pruned (FIT-002).
    prune_min_activity: float = 0.0
    prune_keep_min: int = 16
    split_every: int | None = None
    split_count: int = 0
    split_mode: str = "duplicate"      # duplicate/fp/support/residual/ranked/absgrad/freq modes
    split_scale: float = 0.7
    split_oversample: float = 1.0       # residual_add candidate multiplier before spacing NMS
    split_min_spacing: float = 0.0      # residual_add NMS radius = this * base densify scale
    split_color_init: str = "target"    # target or residual; additive renderers force residual
    absgrad_decay: float = 1.0          # AbsGS-style |dL/dmu| accumulation decay
    relocate_every: int | None = None
    relocate_at_split: bool = False      # relocate on split/growth iterations without a separate timer
    relocate_count: int = 0
    relocate_init_opacity: float = 0.05  # low-alpha function-preserving warm start
    relocate_residual_downsample: int = 1  # coarse residual candidate search; 1 = exact full-res
    # residual_tensor_add anisotropy, mirroring InitConfig semantics so the densifier and the
    # init agree on what anisotropy means: ratio = 1 + (max_axis_ratio-1)*coherence**power.
    densify_max_axis_ratio: float = 6.0
    densify_coherence_power: float = 1.0
    max_gaussians: int | None = None
    adaptive_count: bool = False           # grow until target/stall/max-N instead of fixed-N
    target_ms_ssim: float | None = None    # adaptive stop target; final metric is always recorded
    target_bpp: float | None = None        # raw-attribute rate cap/target used by adaptive mode
    adaptive_growth_every: int = 50
    adaptive_growth_count: int = 64
    adaptive_split_mode: str = "residual_tensor_add"
    adaptive_min_delta_psnr: float = 0.02  # min gain between adaptive checks to keep growing
    adaptive_patience: int = 2             # stale adaptive checks before stopping
    early_stop_patience: int | None = None  # logged evals without improvement; None disables
    early_stop_min_delta: float = 0.0       # PSNR improvement required to reset patience
    early_stop_min_iters: int = 0           # do not early-stop before this iteration

    def __post_init__(self):
        # aa_dilation adds Sigma + d*I; a negative value yields negative inverse variances and
        # NaN renders (CORE-004). Reject it at construction rather than mid-fit.
        if self.aa_dilation < 0.0:
            raise ValueError(f"aa_dilation must be >= 0, got {self.aa_dilation}")
        if self.color_solve_every is not None and self.color_solve_every < 0:
            raise ValueError(
                f"color_solve_every must be >= 0 or None, got {self.color_solve_every}")
        if self.color_solve_lambda < 0.0:
            raise ValueError(
                f"color_solve_lambda must be >= 0, got {self.color_solve_lambda}")
        if self.color_solve_maxiter <= 0:
            raise ValueError(
                f"color_solve_maxiter must be > 0, got {self.color_solve_maxiter}")
        if self.color_basis not in ("constant", "affine"):
            raise ValueError(
                f"color_basis must be constant or affine, got {self.color_basis!r}")
        if self.color_grad_l2 < 0.0:
            raise ValueError(f"color_grad_l2 must be >= 0, got {self.color_grad_l2}")
        if self.ssim_backend not in ("builtin", "fused", "auto"):
            raise ValueError(
                f"ssim_backend must be builtin, fused, or auto, got {self.ssim_backend!r}")
        if self.split_oversample < 1.0:
            raise ValueError(f"split_oversample must be >= 1, got {self.split_oversample}")
        if self.split_min_spacing < 0.0:
            raise ValueError(f"split_min_spacing must be >= 0, got {self.split_min_spacing}")
        if self.split_color_init not in ("target", "residual"):
            raise ValueError(
                f"split_color_init must be target or residual, got {self.split_color_init!r}")
        if not 0.0 <= self.absgrad_decay <= 1.0:
            raise ValueError(f"absgrad_decay must be in [0, 1], got {self.absgrad_decay}")
        if self.relocate_count < 0:
            raise ValueError(f"relocate_count must be >= 0, got {self.relocate_count}")
        if not 0.0 < self.relocate_init_opacity < 1.0:
            raise ValueError(
                f"relocate_init_opacity must be in (0, 1), got {self.relocate_init_opacity}")
        if self.relocate_residual_downsample < 1:
            raise ValueError(
                "relocate_residual_downsample must be >= 1, "
                f"got {self.relocate_residual_downsample}")
        if self.max_gaussians is not None and self.max_gaussians <= 0:
            raise ValueError(f"max_gaussians must be > 0 when set, got {self.max_gaussians}")
        if self.target_ms_ssim is not None and not 0.0 <= self.target_ms_ssim <= 1.0:
            raise ValueError(f"target_ms_ssim must be in [0, 1], got {self.target_ms_ssim}")
        if self.target_bpp is not None and self.target_bpp <= 0.0:
            raise ValueError(f"target_bpp must be > 0 when set, got {self.target_bpp}")
        adaptive_modes = (
            "residual_add", "residual_tensor_add", "ranked_wave", "freq_violation",
            "fp_duplicate", "moment_preserving", "support_duplicate",
        )
        if self.adaptive_split_mode not in adaptive_modes:
            raise ValueError(
                "adaptive_split_mode must be one of "
                f"{', '.join(adaptive_modes)}, got {self.adaptive_split_mode!r}")
        if self.adaptive_growth_every <= 0:
            raise ValueError(
                f"adaptive_growth_every must be > 0, got {self.adaptive_growth_every}")
        if self.adaptive_growth_count <= 0:
            raise ValueError(
                f"adaptive_growth_count must be > 0, got {self.adaptive_growth_count}")
        if self.adaptive_min_delta_psnr < 0.0:
            raise ValueError(
                f"adaptive_min_delta_psnr must be >= 0, got {self.adaptive_min_delta_psnr}")
        if self.adaptive_patience <= 0:
            raise ValueError(f"adaptive_patience must be > 0, got {self.adaptive_patience}")
        if self.adaptive_count and self.max_gaussians is None and self.target_bpp is None:
            raise ValueError("adaptive_count requires max_gaussians or target_bpp")


@dataclass
class PyramidConfig:
    levels: int = 4
    # fraction of the total budget placed at each level (coarse -> fine). Need not sum to 1;
    # normalized internally and placed by largest-remainder so level budgets sum exactly to
    # num_gaussians (HIER-002).
    level_fractions: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    # A cosine lr_schedule spans the whole pyramid run (one decay across all levels), not a
    # per-level warm restart, so it is comparable to a single-stage cosine (HIER-002 / ADR-0010).
    iters_per_level: int = 500
    residual_grad_sigma: float = 0.8   # structure tensor of the residual is sharper
    level_grad_sigmas: list[float] | None = None
    level_tensor_sigmas: list[float] | None = None
    evaluate_prefixes: bool = True


@dataclass
class GenConfig:
    """Configuration for text-to-Gaussian generation via latent SDS (GEN-001)."""

    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    height: int = 512
    width: int = 512
    num_gaussians: int = 5000
    sample_steps: int = 30
    fit_iters: int = 500
    refine_steps: int = 200
    sample_guidance_scale: float = 7.5
    sds_guidance_scale: float = 30.0
    negative_prompt: str = ""
    seed: int = 0
    dtype: str = "float16"             # float32, float16, bfloat16
    device: str | None = None
    init_strategy: str = "aniso_flanking"
    init_opacity: float = 0.8
    renderer: str = "additive"
    render_chunk: int = 512
    sigma_cutoff: float = 3.0
    support_fade: bool = False
    aa_dilation: float = 0.0
    lr_means: float = 1e-2
    lr_scales: float = 3e-3
    lr_rot: float = 1e-3
    lr_color: float = 1e-2
    lr_opacity: float = 3e-3
    sds_t_min: int = 20
    sds_t_max: int = 980
    sds_weight: float = 1.0
    opacity_l1: float = 1e-4
    scale_l2: float = 1e-5
    color_l2: float = 0.0
    grad_clip_norm: float | None = 1.0
    min_scale: float = 0.25
    max_scale: float | None = None
    color_min: float = 0.0
    color_max: float = 1.0
    save_resolutions: list[int] = field(default_factory=lambda: [256, 512, 1024])

    def __post_init__(self):
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"height/width must be positive, got {self.height}x{self.width}")
        if self.height % 8 != 0 or self.width % 8 != 0:
            raise ValueError(
                f"height and width must be divisible by 8 for Stable Diffusion, "
                f"got {self.height}x{self.width}"
            )
        if self.num_gaussians <= 0:
            raise ValueError(f"num_gaussians must be > 0, got {self.num_gaussians}")
        if self.sample_steps <= 0:
            raise ValueError(f"sample_steps must be > 0, got {self.sample_steps}")
        if self.fit_iters < 0:
            raise ValueError(f"fit_iters must be >= 0, got {self.fit_iters}")
        if self.refine_steps < 0:
            raise ValueError(f"refine_steps must be >= 0, got {self.refine_steps}")
        if self.sds_t_min < 0 or self.sds_t_max <= self.sds_t_min:
            raise ValueError(
                f"expected 0 <= sds_t_min < sds_t_max, got "
                f"{self.sds_t_min}..{self.sds_t_max}"
            )
        if not 0.0 < self.init_opacity < 1.0:
            raise ValueError(f"init_opacity must be in (0, 1), got {self.init_opacity}")
        if self.min_scale <= 0.0:
            raise ValueError(f"min_scale must be > 0, got {self.min_scale}")
        if self.max_scale is not None and self.max_scale <= self.min_scale:
            raise ValueError(
                f"max_scale must be > min_scale, got {self.max_scale} <= {self.min_scale}"
            )
