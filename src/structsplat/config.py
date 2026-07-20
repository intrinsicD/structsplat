"""Typed configuration objects. Pure-Python (no torch), safe to import anywhere."""
from __future__ import annotations
from dataclasses import dataclass, field
import math


DEFAULT_INIT_STRATEGY = "quadtree_wse"
DEFAULT_PREDICTOR_FALLBACK_STRATEGY = DEFAULT_INIT_STRATEGY

REFINE_SITES = (
    "none", "residual", "residual_tensor", "support", "responsibility", "absgrad",
    "ranked", "freq_violation",
)
REFINE_PRIMITIVES = ("duplicate", "fp", "moment_preserving", "sampled_add")
REFINE_NMS_MODES = ("off", "on")

_REFINE_ALIAS_TABLE: dict[str, tuple[str, str, str]] = {
    "none": ("none", "duplicate", "off"),
    "duplicate": ("residual", "duplicate", "off"),
    "fp_duplicate": ("residual", "fp", "off"),
    "moment_preserving": ("residual", "moment_preserving", "off"),
    "support_duplicate": ("support", "duplicate", "off"),
    "residual_add": ("residual", "sampled_add", "off"),
    "residual_tensor_add": ("residual_tensor", "sampled_add", "off"),
    "ranked_wave": ("ranked", "fp", "off"),
    "absgrad_wave": ("absgrad", "fp", "off"),
    "freq_violation": ("freq_violation", "fp", "off"),
    "residual_add_nms": ("residual", "sampled_add", "on"),
    "residual_tensor_add_nms": ("residual_tensor", "sampled_add", "on"),
}

_FACTORED_REFINE_ALIASES: dict[tuple[str, str], str] = {
    ("none", "duplicate"): "none",
    ("residual", "duplicate"): "duplicate",
    ("residual", "fp"): "fp_duplicate",
    ("residual", "moment_preserving"): "moment_preserving",
    ("support", "duplicate"): "support_duplicate",
    ("residual", "sampled_add"): "residual_add",
    ("residual_tensor", "sampled_add"): "residual_tensor_add",
    ("ranked", "fp"): "ranked_wave",
    ("absgrad", "fp"): "absgrad_wave",
    ("freq_violation", "fp"): "freq_violation",
}


def refine_alias_to_axes(mode: str) -> tuple[str, str, str]:
    try:
        return _REFINE_ALIAS_TABLE[mode]
    except KeyError as exc:
        expected = ", ".join(sorted(_REFINE_ALIAS_TABLE))
        raise ValueError(f"unknown refine alias {mode!r}; expected one of: {expected}") from exc


def refine_axes_to_alias(site: str, primitive: str, nms: str = "off") -> str:
    if site == "none":
        return "none"
    alias = _FACTORED_REFINE_ALIASES.get((site, primitive))
    if alias is None:
        return f"{site}_{primitive}"
    if nms == "on" and alias in ("residual_add", "residual_tensor_add"):
        return f"{alias}_nms"
    return alias


def default_flank_offset_for_strategy(strategy: str) -> float:
    return 0.5 if strategy == "aniso_flanking" else 0.0


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
    strategy: str = DEFAULT_INIT_STRATEGY  # see structsplat.init.STRATEGIES
    num_gaussians: int = 20000
    candidate_oversample: float = 6.0  # WSE draws oversample*N candidates
    density_base: float = 0.05         # floor so flat regions still get some coverage
    density_power: float = 1.0         # density ~ energy**power
    density_mode: str = "structure"    # structure, gradient, variance, hybrid, or uniform
    # wse, density_random, floyd_steinberg, jittered_grid, dart_throwing, halton,
    # farthest_point, or cvt
    sampling_mode: str = "wse"
    # Terminal-set-preserving recursive WSE order; opt-in until INIT-009's prefix audit promotes it.
    wse_progressive_order: bool = False
    # anisotropy: axis ratio cap for edge Gaussians (major/minor); 1.0 => isotropic
    max_axis_ratio: float = 6.0
    coherence_power: float = 1.0     # maps coherence -> anisotropy; >1 is more conservative
    orientation_mode: str = "tensor"   # tensor (strategy default), random, or zero
    scale_mode: str = "spacing"        # spacing, uniform, or knn
    init_scale_mult: float = 1.0       # multiply local spacing to get initial std
    scale_cap_mode: str = "none"       # none, hard, feature, or feature_rel
    scale_cap_max: float | None = None # absolute sigma cap, in pixels
    scale_feature_sigma: float = 3.0   # feature mode: visible half-length ~= sigma*value
    scale_feature_min: float = 0.75    # minimum adaptive sigma cap
    scale_feature_energy_frac: float = 0.25  # stop edge run when energy drops below this local frac
    # feature_rel mode: cap detail Gaussians relative to local WSE radius / quadtree leaf side.
    # RS axes are (along edge, across edge), so along edges needs a looser cap.
    scale_feature_rel_across: float = 1.0
    scale_feature_rel_along: float = 4.0
    scale_feature_rel_min: float = 0.5
    scale_feature_rel_quantile: float = 0.4
    scale_feature_rel_flat_mult: float | None = None  # None leaves sparse/flat rows uncapped
    background_fraction: float = 0.0  # fraction of total N reserved for frozen broad bg rows
    background_grid: int = 0          # max side of jittered-grid bg layer; 0 disables it
    flank_offset_frac: float | None = None  # strategy-aware default; override to ablate offset
    color_mode: str = "bilinear"       # bilinear, local_mean, two_sided, or aggregate (quadtree)
    color_radius: float = 1.5          # local mean radius or extra side-sample offset
    opacity_mode: str = "none"         # none or constant
    init_opacity: float = 0.9
    # FF-001: feed-forward warm-start entry point. The first implementation supports a stable
    # predictor interface, optional saved-field warm starts, and a deterministic tensor-prior
    # fallback while learned predictors/training mature.
    predictor_checkpoint: str | None = None
    predictor_fallback_strategy: str = DEFAULT_PREDICTOR_FALLBACK_STRATEGY
    seed: int = 0

    def __post_init__(self):
        if self.flank_offset_frac is None:
            self.flank_offset_frac = default_flank_offset_for_strategy(self.strategy)
        if self.flank_offset_frac < 0.0:
            raise ValueError(f"flank_offset_frac must be >= 0, got {self.flank_offset_frac}")
        # WSE draws candidate_oversample * N candidates then reduces to N; < 1 cannot supply N
        # candidates and silently breaks the exact-N contract (INIT-005).
        if self.candidate_oversample < 1.0:
            raise ValueError(
                f"candidate_oversample must be >= 1, got {self.candidate_oversample}")
        if self.wse_progressive_order:
            pure_wse = self.strategy == "quadtree_wse" or (
                self.strategy in ("iso_blue_noise", "aniso_onedge", "aniso_flanking")
                and self.sampling_mode == "wse"
            )
            if not pure_wse:
                raise ValueError(
                    "wse_progressive_order requires a pure-WSE layout: quadtree_wse, or "
                    "iso_blue_noise/aniso_onedge/aniso_flanking with sampling_mode='wse'; "
                    f"got strategy={self.strategy!r}, sampling_mode={self.sampling_mode!r}")
        if self.scale_cap_mode not in ("none", "hard", "feature", "feature_rel"):
            raise ValueError(
                "scale_cap_mode must be none, hard, feature, or feature_rel, "
                f"got {self.scale_cap_mode!r}")
        if self.scale_cap_max is not None and self.scale_cap_max <= 0.0:
            raise ValueError(f"scale_cap_max must be > 0 when set, got {self.scale_cap_max}")
        for name in (
            "scale_feature_sigma",
            "scale_feature_min",
            "scale_feature_rel_across",
            "scale_feature_rel_along",
            "scale_feature_rel_min",
        ):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if not (0.0 <= self.scale_feature_energy_frac <= 1.0):
            raise ValueError(
                "scale_feature_energy_frac must be in [0, 1], "
                f"got {self.scale_feature_energy_frac}")
        if not (0.0 <= self.scale_feature_rel_quantile <= 1.0):
            raise ValueError(
                "scale_feature_rel_quantile must be in [0, 1], "
                f"got {self.scale_feature_rel_quantile}")
        if self.scale_feature_rel_flat_mult is not None and self.scale_feature_rel_flat_mult <= 0.0:
            raise ValueError(
                "scale_feature_rel_flat_mult must be > 0 when set, "
                f"got {self.scale_feature_rel_flat_mult}")
        if not (0.0 <= self.background_fraction < 1.0):
            raise ValueError(
                "background_fraction must be in [0, 1), "
                f"got {self.background_fraction}")
        if self.background_grid < 0:
            raise ValueError(f"background_grid must be >= 0, got {self.background_grid}")
        if self.background_fraction > 0.0 and self.background_grid <= 0:
            raise ValueError(
                "background_grid must be > 0 when background_fraction is enabled")
        if self.predictor_fallback_strategy == "feedforward":
            raise ValueError("predictor_fallback_strategy cannot be feedforward")


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
    loss_weighting: str = "none"       # none or tensor; weights only the pixel-loss term
    loss_weight_beta: float = 1.0      # tensor mode: w = 1 + beta * normalized tensor energy
    loss_warmup_iters: int = 0
    loss_warmup_pixel_loss: str = "l2"
    # Optional frequency-ordering curriculum. A factor >1 first supervises against an
    # area-downsampled/bilinear-upsampled target, then cosine-blends to the full target over
    # this fraction of the global schedule. Defaults preserve the exact legacy objective.
    loss_target_downsample: int = 1
    loss_target_full_frac: float = 0.0
    ssim_weight: float = 0.3           # loss = (1-w)*L1 + w*(1-SSIM); Instant-GI/AIR default
    geometry_loss_weight: float = 0.0  # Sobel gradient-consistency regularizer; 0 disables
    geometry_loss_every: int = 1       # apply GCR every N steps; active weight scales by N
    ssim_backend: str = "builtin"      # builtin, fused (optional), or auto
    compute_lpips: bool = False        # opt-in: loads a separate AlexNet LPIPS model
    target_psnr: float | None = None   # record iters-to-target if set
    target_psnrs: list[float] = field(default_factory=list)
    log_every: int = 100
    # Opt-in model selection. The best-PSNR policy evaluates the post-transition field at the
    # logging cadence and may restore only a checkpoint with the terminal Gaussian count.
    checkpoint_policy: str = "terminal"  # terminal or best_psnr_final_count
    sigma_cutoff: float = 3.0          # render support radius in std devs
    support_fade: bool = False         # C0 compact support: subtract Gaussian tail at cutoff
    # None preserves support_fade as a static bool. A fraction enables fade only for the early
    # part of the fit, then ramps it off over support_fade_crossfade_iters.
    support_fade_until_frac: float | None = None
    support_fade_crossfade_iters: int = 10
    aa_dilation: float = 0.0           # EWA-style low-pass: render with Sigma + d*I (px^2)
    # GaussianImage++-style birth-cohort covariance filtering. Each cohort receives
    # min(max_variance, H*W/(alpha*N_after)) px^2 of fixed isotropic variance.
    covariance_filter_mode: str = "none"  # none or generation_density
    covariance_filter_alpha: float = 9.0 * math.pi
    covariance_filter_max_variance: float = 300.0
    render_chunk: int = 512            # reference renderer: max(render_chunk,64)*4096 elements
    render_checkpoint: bool = False    # checkpoint reference-render slices to reduce backward memory
    renderer: str = "normalized"       # normalized/additive/cuda/cuda_tiled/gsplat variants
    color_basis: str = "constant"      # constant or affine local color model
    color_grad_l2: float = 1e-4        # affine coefficient L2 regularization
    color_solve_every: int | None = None  # periodic fixed-geometry RGB least-squares solve
    color_solve_schedule: str = "every"   # none/every/init/final/on_split, composable with +
    color_solve_lambda: float = 1e-4       # Tikhonov pull toward pre-solve colors
    color_solve_maxiter: int = 32          # CG iterations for the color normal equations
    qat_mode: str = "off"                  # off, ste, or noise; COMP-004 fit-time QAT
    lambda_rate: float = 0.0               # weight on differentiable in-loop rate proxy
    qat_bits_means: int = 16
    qat_bits_scales: int = 8
    qat_bits_rot: int = 8
    qat_bits_colors: int = 8
    qat_bits_opacity: int = 8
    lr_schedule: str = "none"          # none, step, or cosine
    # Fraction of a cosine schedule held at full LR before decay. Zero preserves the
    # horizon-wide cosine; values near one reserve decay for a final stabilization tail.
    lr_cosine_start_frac: float = 0.0
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
    split_mode: str = "duplicate"      # legacy alias for refine_site/primitive/nms
    # none/residual/residual_tensor/support/responsibility/absgrad/ranked/freq
    refine_site: str | None = None
    refine_primitive: str | None = None  # duplicate/fp/moment_preserving/sampled_add
    refine_nms: str | None = None      # off/on; on applies sampled-add spacing defaults
    split_scale: float = 0.7
    split_oversample: float = 1.0       # residual_add candidate multiplier before spacing NMS
    split_min_spacing: float = 0.0      # residual_add NMS radius = this * base densify scale
    split_schedule_stops_at_max: bool = False  # suppress later split-triggered side effects at cap
    split_color_init: str = "target"    # target or residual; additive renderers force residual
    # sampled-add site score; signed_gaussian is FIT-017's coherent-error hypothesis
    sampled_add_score: str = "legacy_abs"  # legacy_abs, gaussian_abs, or signed_gaussian
    # FIT-018: responsibility-weighted squared error divided by ownership mass**alpha.
    responsibility_mass_alpha: float = 0.7
    seed_new_row_optimizer_state: bool = False  # seed split children from parent/median moments
    new_row_temper_iters: int = 0        # post-insert update ramp length; 0 disables
    new_row_temper_start: float = 0.25   # first-step update multiplier for young rows
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
    adaptive_continue_after_stop: bool = False  # freeze N and optimize after adaptive criterion
    early_stop_patience: int | None = None  # logged evals without improvement; None disables
    early_stop_min_delta: float = 0.0       # PSNR improvement required to reset patience
    early_stop_min_iters: int = 0           # do not early-stop before this iteration

    def __post_init__(self):
        # aa_dilation adds Sigma + d*I; a negative value yields negative inverse variances and
        # NaN renders (CORE-004). Reject it at construction rather than mid-fit.
        if self.aa_dilation < 0.0:
            raise ValueError(f"aa_dilation must be >= 0, got {self.aa_dilation}")
        if not math.isfinite(self.lr_cosine_start_frac) or not (
            0.0 <= self.lr_cosine_start_frac < 1.0
        ):
            raise ValueError(
                "lr_cosine_start_frac must be finite and in [0, 1), "
                f"got {self.lr_cosine_start_frac}"
            )
        if self.checkpoint_policy not in ("terminal", "best_psnr_final_count"):
            raise ValueError(
                "checkpoint_policy must be terminal or best_psnr_final_count, "
                f"got {self.checkpoint_policy!r}"
            )
        if (
            self.checkpoint_policy == "best_psnr_final_count"
            and self.support_fade_until_frac is not None
        ):
            raise ValueError(
                "checkpoint_policy='best_psnr_final_count' does not support scheduled "
                "support fade; checkpoint scores would use different render policies"
            )
        if self.covariance_filter_mode not in ("none", "generation_density"):
            raise ValueError(
                "covariance_filter_mode must be none or generation_density, "
                f"got {self.covariance_filter_mode!r}"
            )
        if not math.isfinite(self.covariance_filter_alpha) or self.covariance_filter_alpha <= 0.0:
            raise ValueError(
                "covariance_filter_alpha must be > 0, "
                f"got {self.covariance_filter_alpha}"
            )
        if (
            not math.isfinite(self.covariance_filter_max_variance)
            or self.covariance_filter_max_variance < 0.0
        ):
            raise ValueError(
                "covariance_filter_max_variance must be >= 0, "
                f"got {self.covariance_filter_max_variance}"
            )
        if self.support_fade_until_frac is not None and not (
            0.0 <= self.support_fade_until_frac <= 1.0
        ):
            raise ValueError(
                "support_fade_until_frac must be in [0, 1] when set, "
                f"got {self.support_fade_until_frac}")
        if self.support_fade_crossfade_iters < 0:
            raise ValueError(
                "support_fade_crossfade_iters must be >= 0, "
                f"got {self.support_fade_crossfade_iters}")
        if self.color_solve_every is not None and self.color_solve_every < 0:
            raise ValueError(
                f"color_solve_every must be >= 0 or None, got {self.color_solve_every}")
        color_solve_tokens = set(str(self.color_solve_schedule).split("+"))
        valid_color_solve_tokens = {"none", "every", "init", "final", "on_split"}
        if not color_solve_tokens <= valid_color_solve_tokens or (
            "none" in color_solve_tokens and len(color_solve_tokens) > 1
        ):
            raise ValueError(
                "color_solve_schedule must be none, every, init, final, on_split, "
                f"or a + composition, got {self.color_solve_schedule!r}")
        if (
            self.checkpoint_policy == "best_psnr_final_count"
            and "final" in color_solve_tokens
        ):
            raise ValueError(
                "checkpoint_policy='best_psnr_final_count' does not support a final-only "
                "color solve; restoring an earlier checkpoint would bypass that solve"
            )
        if self.color_solve_lambda < 0.0:
            raise ValueError(
                f"color_solve_lambda must be >= 0, got {self.color_solve_lambda}")
        if self.color_solve_maxiter <= 0:
            raise ValueError(
                f"color_solve_maxiter must be > 0, got {self.color_solve_maxiter}")
        if self.qat_mode not in ("off", "ste", "noise"):
            raise ValueError(f"qat_mode must be off, ste, or noise, got {self.qat_mode!r}")
        if self.loss_weighting not in ("none", "tensor"):
            raise ValueError(
                f"loss_weighting must be none or tensor, got {self.loss_weighting!r}")
        if self.loss_weight_beta < 0.0:
            raise ValueError(f"loss_weight_beta must be >= 0, got {self.loss_weight_beta}")
        if (
            isinstance(self.loss_target_downsample, bool)
            or not isinstance(self.loss_target_downsample, int)
            or self.loss_target_downsample < 1
        ):
            raise ValueError(
                "loss_target_downsample must be an integer >= 1, "
                f"got {self.loss_target_downsample!r}"
            )
        if not math.isfinite(self.loss_target_full_frac):
            raise ValueError(
                "loss_target_full_frac must be finite, "
                f"got {self.loss_target_full_frac}"
            )
        if self.loss_target_downsample == 1:
            if self.loss_target_full_frac != 0.0:
                raise ValueError(
                    "loss_target_full_frac must be 0 when loss_target_downsample=1"
                )
        elif not 0.0 < self.loss_target_full_frac <= 1.0:
            raise ValueError(
                "loss_target_full_frac must be in (0, 1] when "
                f"loss_target_downsample>1, got {self.loss_target_full_frac}"
            )
        if not math.isfinite(self.geometry_loss_weight) or self.geometry_loss_weight < 0.0:
            raise ValueError(
                "geometry_loss_weight must be finite and >= 0, "
                f"got {self.geometry_loss_weight}"
            )
        if self.geometry_loss_every <= 0:
            raise ValueError(
                "geometry_loss_every must be > 0, "
                f"got {self.geometry_loss_every}"
            )
        if self.loss_target_downsample > 1 and self.geometry_loss_weight > 0.0:
            raise ValueError(
                "loss-target curriculum does not yet support geometry_loss_weight>0; "
                "the scheduled gradient-target semantics must be defined first"
            )
        effective_color_solve_tokens = set(color_solve_tokens)
        effective_color_solve_tokens.discard("none")
        if self.color_solve_every is None or self.color_solve_every <= 0:
            effective_color_solve_tokens.discard("every")
        if self.loss_target_downsample > 1 and effective_color_solve_tokens:
            raise ValueError(
                "loss-target curriculum does not yet support color solving; "
                "the scheduled solve-target semantics must be defined first"
            )
        if self.lambda_rate < 0.0:
            raise ValueError(f"lambda_rate must be >= 0, got {self.lambda_rate}")
        for name in (
            "qat_bits_means", "qat_bits_scales", "qat_bits_rot",
            "qat_bits_colors", "qat_bits_opacity",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.color_basis not in ("constant", "affine"):
            raise ValueError(
                f"color_basis must be constant or affine, got {self.color_basis!r}")
        if self.color_grad_l2 < 0.0:
            raise ValueError(f"color_grad_l2 must be >= 0, got {self.color_grad_l2}")
        if self.ssim_backend not in ("builtin", "fused", "auto"):
            raise ValueError(
                f"ssim_backend must be builtin, fused, or auto, got {self.ssim_backend!r}")
        if not math.isfinite(self.prune_min_activity) or self.prune_min_activity < 0.0:
            raise ValueError(
                "prune_min_activity must be finite and >= 0, "
                f"got {self.prune_min_activity}"
            )
        has_factored_refine = (
            self.refine_site is not None
            or self.refine_primitive is not None
            or self.refine_nms is not None
        )
        if has_factored_refine and self.split_mode not in _REFINE_ALIAS_TABLE:
            legacy_site, legacy_primitive, legacy_nms = ("residual", "duplicate", "off")
        else:
            legacy_site, legacy_primitive, legacy_nms = refine_alias_to_axes(self.split_mode)
        site = self.refine_site or legacy_site
        primitive = self.refine_primitive or legacy_primitive
        nms = self.refine_nms or legacy_nms
        if site not in REFINE_SITES:
            raise ValueError(
                f"refine_site must be one of {', '.join(REFINE_SITES)}, got {site!r}")
        if primitive not in REFINE_PRIMITIVES:
            raise ValueError(
                "refine_primitive must be one of "
                f"{', '.join(REFINE_PRIMITIVES)}, got {primitive!r}")
        if nms not in REFINE_NMS_MODES:
            raise ValueError(
                f"refine_nms must be off or on, got {nms!r}")
        self.refine_site = site
        self.refine_primitive = primitive
        self.refine_nms = nms
        self.split_mode = refine_axes_to_alias(site, primitive, nms)
        if not math.isfinite(self.responsibility_mass_alpha) or not (
            0.0 < self.responsibility_mass_alpha <= 1.0
        ):
            raise ValueError(
                "responsibility_mass_alpha must be finite and in (0, 1], "
                f"got {self.responsibility_mass_alpha}"
            )
        if self.refine_site == "responsibility" and self.renderer not in (
            "normalized", "cuda", "cuda_normalized", "cuda_tiled", "cuda_tiled_normalized",
        ):
            raise ValueError(
                "refine_site='responsibility' requires normalized renderer semantics, "
                f"got renderer={self.renderer!r}"
            )
        if self.refine_nms == "on":
            if self.split_min_spacing == 0.0:
                self.split_min_spacing = 1.0
            if self.split_oversample == 1.0:
                self.split_oversample = 8.0
        if self.split_oversample < 1.0:
            raise ValueError(f"split_oversample must be >= 1, got {self.split_oversample}")
        if self.split_min_spacing < 0.0:
            raise ValueError(f"split_min_spacing must be >= 0, got {self.split_min_spacing}")
        if self.split_color_init not in ("target", "residual"):
            raise ValueError(
                f"split_color_init must be target or residual, got {self.split_color_init!r}")
        if self.new_row_temper_iters < 0:
            raise ValueError(
                f"new_row_temper_iters must be >= 0, got {self.new_row_temper_iters}")
        if not 0.0 <= self.new_row_temper_start <= 1.0:
            raise ValueError(
                "new_row_temper_start must be in [0, 1], "
                f"got {self.new_row_temper_start}")
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
        if self.sampled_add_score not in ("legacy_abs", "gaussian_abs", "signed_gaussian"):
            raise ValueError(
                "sampled_add_score must be legacy_abs, gaussian_abs, or signed_gaussian, "
                f"got {self.sampled_add_score!r}")
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
    # Optional explicit per-level iteration counts, coarse -> fine. When set, this overrides
    # iters_per_level and lets HIER-004 test schedules like 300/1200 at the same total horizon.
    level_iters: list[int] | None = None
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
    init_strategy: str = DEFAULT_INIT_STRATEGY
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
