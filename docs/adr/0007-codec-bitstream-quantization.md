# ADR-0007: Codec bitstream — per-attribute uniform quantization, Morton-delta positions, zlib

## Context
COMP-001 needs a first rate-distortion point: turn a fitted `GaussianField` into a compact
bitstream and measure bpp vs PSNR/MS-SSIM. GaussianImage-family codecs combine per-attribute
quantization (fp16 positions, ~6-10 bit covariance, 6-8 bit or vector-quantized colors) with
quantization-aware fine-tuning and an entropy stage. We want the simplest codec that (a) produces
honest bpp numbers for the benchmark, (b) exploits what is specific to this representation, and
(c) leaves room for a learned entropy model later.

## Decision
`codec.py` encodes attributes as independent planar streams, each uniformly quantized and
zlib-compressed:
- **means**: fixed-point over the image extent (default 16 bit/coordinate). Because the
  normalized renderer is **order-independent** (ADR-0003), Gaussians are first reordered along a
  Morton curve and positions are stored as wraparound-safe deltas — near-sorted small integers
  that deflate well.
- **log_scales**: uniform over the fitter's clamp range `[log 0.35, log max(H, W)]`
  (default 8 bit).
- **rotation**: canonicalized to `[0, pi)` — a 2D Gaussian is invariant under `theta + pi` —
  then uniform (default 8 bit).
- **colors**: uniform over per-channel `[min, max]` stored in the header (colors are unbounded
  since opacity is folded in; a fixed [0,1] range would clip them). Default 8 bit/channel.

`qat_finetune` runs a short straight-through-estimator fine-tune through the quantized renderer
before encoding; color ranges are frozen at its start (a moving quantization range defeats
convergence) and travel in the returned config. The decoded field renders with the same
normalized reference renderer — no separate decode path to maintain.

## Consequences
+ Actual-bitstream bpp (header + streams), not a theoretical estimate; wired into
  `benchmarks/rate_distortion.py` as the RD protocol for COMP-001.
+ Morton-delta positions are the representation-specific win: blue-noise positions carry no
  exploitable order, but order-independence means we may impose one for free.
+ QAT recovers most coarse-bit loss (STE, per test_codec.py) without touching `fit.py`.
- Uniform scalar quantization + zlib is weaker than learned entropy models or VQ (GaussianImage
  reports ~56 bits/Gaussian with residual VQ colors); acceptable for a reference codec, and the
  stream layout keeps those upgrades local to `codec.py`.
- The pyramid's LOD-prefix property is not yet exploited (progressive decoding would need
  per-level streams); noted in COMP-001 as follow-up.

## Amendment (COMP-002): self-describing header + fitted-domain means

The bitstream is still `SSPL1`, but the JSON header gained fields (backward-compatible: `decode`
supplies pre-COMP-002 defaults when they are absent):
- **means domain**: `means_lo`/`means_hi` — the image box unioned with the fitted means' actual
  range. `fit()` never clamps means, so an off-image Gaussian used to snap to the border with
  error unbounded by the lattice step; quantizing over the stored extent bounds it by one step.
- **render semantics**: `renderer`, `aa_dilation`, `sigma_cutoff`, `render_chunk`. The blob is now
  self-describing — `decode_and_render(blob)` reproduces the fitted renderer's output with no
  out-of-band `FitConfig`. `qat_finetune`/`rd_point` render through `fcfg.renderer` (not a
  hardcoded normalized path), so additive/CUDA/gsplat-fit fields settle and are scored under their
  own compositing model. `scale_max` is intentionally **not** stored: it is a fit-time
  optimization cap, irrelevant to the frozen decoded field.

RD metrics are display-referred (render clamped to `[0,1]` before PSNR/MS-SSIM); the equal-budget
no-STE control row in `rate_distortion.py` isolates QAT's lattice-settling from the extra compute
it spends. See ADR-0006 for the additive renderer these paths now honor.

## Links
Depends on ADR-0002 (RS params), ADR-0003 (order-independent normalized renderer). Implements the
first acceptance criterion of COMP-001; COMP-002 fixes render-semantics coverage and the means
domain.
