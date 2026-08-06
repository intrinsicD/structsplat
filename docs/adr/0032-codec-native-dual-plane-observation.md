# ADR-0032: Codec-native dual-plane observation packets

- Status: proposed, implemented default-off
- Date: 2026-08-06
- Task: CORE-016
- Related: ADR-0006, CORE-013, BENCH-019, BENCH-020, BENCH-025, HIER-005--009

## Decision

Add a self-contained, versioned experimental packet that stores a charged conventional image-codec
payload for appearance and an independent sparse nonnegative Field V2 structural measure for
2D-to-3D proposals. Decode the appearance payload into signed cardinal-prefiltered coefficients of
a finite normalized Gaussian lattice, and expose the two planes to realtime-gs only as a required
pair: `GaussianObservationField` structure plus an `ObservationQueryBackend` appearance teacher.

The packet and adapter remain absent from `pipeline.run_pipeline`, `scripts/convert.py`, the native
NPZ/SSPL1 formats, and every default dispatch.

## Context

The HIER-005--009 contraction experiments repeatedly couple four jobs in every explicit row:
appearance, reconstruction weight, spatial support, and 2D-to-3D proposal structure. Reducing row
count then changes all four at once and has produced holes, grids, rings, or redistributed local
error even when average distortion improves. Realtime-gs already accepts a pluggable observation
query backend, so it does not require every appearance sample to be persisted as a lifted Gaussian.

The codec-native alternative makes the ownership boundary explicit:

- the appearance plane owns color fidelity and continuous queries;
- the structural plane owns proposal density, anisotropy, and nonnegative mass;
- the packet owns every byte required for cold reconstruction; and
- the adapter owns coordinate and query parity at the realtime-gs seam.

## Representation contract

For decoded samples `Y`, a separable finite Gaussian kernel `K`, and its boundary normalizers
`d_y = K_y 1`, `d_x = K_x 1`, the decoder deterministically derives signed coefficients `C` from

```text
K_y C K_x^T = Y * d_y * d_x^T.
```

Bounded Jacobi axis solves are allowed only when the finite kernel is strictly diagonally dominant.
The resulting normalized Gaussian query interpolates decoded pixel centers while remaining smooth
between them. `C` is derived decoder state: it is hashed in the manifest but is not an uncharged
payload. Coefficients may leave `[0,1]`; clipping is a display operation, never part of the field
equation.

The structural plane is an exact-count `ObservationField2D` with zero RGB coefficients and an
independent nonnegative mass channel. Its structure-tensor density and anisotropy are deterministic
given the embedded config and seed. A structural field without the paired appearance backend is not
a faithful teacher.

## Consequences

Positive consequences:

- appearance rate is priced by complete physical packet bytes rather than explicit-row proxies;
- high-frequency pixel-center replay does not require a nonlinear per-image Gaussian fit;
- appearance quality and proposal count can be varied independently; and
- the base packet remains NumPy/Pillow-only while torch/realtime-gs imports stay optional and lazy.

Costs and risks:

- cold decode materializes a full decoded raster and signed coefficient raster;
- the reference query is a finite `(2r+1)^2` gather and is not a production GPU texture kernel;
- cardinal prefiltering can ring between samples and escape the local or global sample envelope;
- a sparse structural measure can be insufficient for downstream geometry even when 2D queries are
  exact; and
- source-file ratios can be misleading when the source is a full frame but the packet stores only a
  crop. Crop-local canonical PNG and bits-per-pixel are mandatory context.

## Evidence boundary

The exposed C0001 development diagnostic selected lossless WebP, sigma `0.45`, radius `3`, eight
prefilter steps, and 512 structural rows after development sweeps. It demonstrates packet plumbing,
pixel-center replay, complete-byte accounting, and synthetic realtime-gs lift compatibility.

A later exposed 23-train/three-reporting-view assay carries the paired backend through real CUDA
CompactCarve and common 3DGS refinement. With a common 10,000-Gaussian cap, the quality-92 WebP /
512-structure-per-view candidate uses 956,301 complete input bytes versus 3,850,647 for the extant
RTGSV containers and exceeds the control on reporting PSNR, MS-SSIM, LPIPS, and alpha IoU. It also
reaches the control's lower terminal PSNR earlier, but its own complete lift/training path is not
faster, uses more peak VRAM, and retains soft halos, fine-detail blur, and occasional floaters.
Stronger exact-mask training and a fixed-topology late polish trade PSNR/gradient fidelity for alpha
and are rejected relative to the simpler matched-cap run.

These reused, reduced-resolution reporting views establish development utility only. They do not
establish held-out/full-resolution compression, continuous-scene fidelity, artifact-free real
multiview reconstruction, final-3D storage savings, or general faster convergence. Exact numbers,
integrity checks, and corrections are in
[`2026-08-06-codec-native-dual-plane-results-audit.md`](../research/2026-08-06-codec-native-dual-plane-results-audit.md).

## Reversal

CORE-017 subsequently composes the same packet into a default-off visibility-ordered alpha-shell
lift. It changes neither this decision nor the packet grammar: structure proposes rays, packet alpha
supplies placement support, appearance supplies radiance, and optional surface cover changes only
covariance/opacity. Its exposed fixed-5k factorial passes the numerical gate but fails mandatory
native review because trailing smear/double silhouettes and blur remain. It is therefore retained
as causal diagnostic evidence and does not authorize a variable-topology, maintained, or default
path.

Delete `codec_native_field.py`, `realtime_gs_adapter.py`, the task-local diagnostic and tests, and
retire CORE-016. No maintained packet, renderer, pipeline, semantic default, or realtime-gs checkout
must be migrated.
