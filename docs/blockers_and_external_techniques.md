# Current blockers and external techniques

**Updated:** 2026-07-13. This replaces the 2026-07-02 recommendation to tune residual
densification: that family has now been extensively screened, the exact CUDA renderer exists, and
the main blocker has moved from local optimizer quality to actual-rate scientific validity.

## Current signal

- Structured placement remains a useful bounded result: the flanking hypothesis is retired,
  quadtree-WSE is the shipped high-budget PSNR choice, and tensor/on-edge placement remains a
  low-budget/MS-SSIM control.
- The completed 168 KiB report is a strong optimizer/policy audit but an invalid compression
  operating point: 71.68–81.15 analytical bpp and about 22 SSPL1 bpp on small prepared images.
- Cosine LR and final color solve produce large float-field endpoint gains in that overcomplete
  lane. This says the optimizer was not converged under the historical default; it does not
  establish a better coded representation.
- Native Image-GS, GaussianImage++, GaussianImage, and AIR evidence is now executable and
  provenance-aware, but still limited to small/bounded or mismatched protocols.
- Current literature directly occupies broad structure-aware allocation, normalized ownership,
  progressive coding, learned initialization, boundary gating, and clustered quantization.

## Blocking issues

### 1. No held-out actual-rate decision benchmark

The current RD script emits real SSPL1 bytes but sweeps counts/bit mixes rather than targeting byte
caps with equal encoder search. The fixed-storage lane normalizes neither bytes by original pixels
nor capacity by rate. BENCH-007 must add:

- target rates 0.25/0.5/1/2/4 bpp;
- complete-stream bytes and original-pixel denominators;
- independently fitted candidate counts with equal QAT/bit-mix search;
- cold-decode validation;
- robust nondominated curves and no-extrapolation BD-rate;
- held-out DIV2K validation, with Kodak only as a replication set.

Until then, no compression ranking is decision-grade.

### 2. The closest direct handcrafted baseline is missing

[Structure-Guided Allocation](https://arxiv.org/abs/2512.24018) already couples SLIC/Sobel
structure classes to allocation, geometry regularization, and adaptive covariance precision. A
random/grid/GaussianImage analogue is no longer the strongest control for the tensor/WSE claim.
BENCH-007 needs a mechanism-faithful common-renderer SLIC/Sobel arm, and BENCH-005 should run the
official method if code is available.

### 3. Count, parameter BPP, and actual rate are mixed across the field

SSPL1 bytes, a 32-byte float payload, Image-GS analytical bits, GaussianImage's
`56N+1728` no-EC formula, checkpoint size, and AIR/native quantized rate answer different
questions. Every report and table must use separate columns, null unavailable quantities, and
forbid substitutions. Header, ranges, codebooks, masks, base layers, and side information count.

### 4. Common controls and native methods answer different questions

Paper-inspired rows inside StructSplat isolate mechanisms under one renderer/fitter. Native
executions preserve external validity but confound initialization, representation, loss,
optimizer, schedule, and code. BENCH-008 should cross fields and renderers only if BENCH-007 shows
a meaningful interaction; native-authentic results remain separate.

### 5. The representation frontier has moved

- [SAD](https://arxiv.org/abs/2604.21984) makes reach and temperature independent and uses
  normalized top-K ownership.
- [SGI](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_SGI_Structured_2D_Gaussians_for_Efficient_and_Compact_Large_Image_CVPR_2026_paper.pdf)
  organizes Gaussians under seeds and entropy-codes them at megapixel scale.
- [WIPES](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_WIPES_Wavelet-based_Visual_Primitives_ICCV_2025_paper.html)
  challenges the Gaussian primitive itself.
- [Contour-Aware 2DGS](https://arxiv.org/abs/2512.23255) directly threatens broad boundary gating.
- [CGVQ](https://arxiv.org/abs/2607.05667) occupies generic clustered codebook quantization.

CORE-007/008 are therefore gated research spikes with these direct controls, not ready-to-build
features.

### 6. Structural regularity does not yet reduce transmitted bits

Progressive WSE improves geometric prefixes, but SSPL1 Morton reordering discards that order and the
codec does not derive geometry at the decoder. COMP-005 is the high-risk follow-up only if
BENCH-007 first proves the structural mechanism at actual rate: derive enhancement geometry
deterministically from an already transmitted base layer and measure whether layout bytes fall
after all side information is counted.

### 7. Native coverage is incomplete

Open native work includes full-resolution/multi-rate GaussianImage++, Image-GS packed streams,
released GaussianImage Kodak/QAT semantics, hardened AIR rate accounting, Instant-GI checkpoint
execution, and current Structure-Guided/SAD/WIPES implementations where available. Missing code
must stay “not run,” not be replaced by a local analogue under the native name.

## Resolved or retired blockers

- **Exact rendering speed:** StructSplat owns an exact CUDA path; the old “PyTorch reference only”
  description is obsolete. Further performance work is PORT-002/003, not a prerequisite for the
  actual-rate killing pilot.
- **Flanking as the flagship:** retired by ABL-006/ADR-0013.
- **Simple residual densification as the immediate answer:** multiple growth, relocation,
  matched-residual, filtering, scale-cap, and recovery interventions have been screened. FIT-017's
  signed-Gaussian score improved the immediate step and then lost after recovery.
- **Progressive WSE prefix correctness:** INIT-009 implements the ordering repair, but codec value
  remains unproved.
- **More high-rate proxy search:** BENCH-006 completed it. Further policy mining at 22+ bpp is not
  the priority.

## Recommended execution order

1. Implement BENCH-007's tested target-rate substrate and freeze its Stage-1 manifest.
2. Run the eight-image 0.5/1.0 bpp killing pilot and apply the preregistered gate.
3. If it passes, run full DIV2K validation plus Kodak replication and expand native actual-RD
   controls.
4. Enter BENCH-008 only for a measured renderer/objective interaction.
5. Enter COMP-005 only when tensor structure survives and explicit layout bytes are the binding
   rate cost.
6. Keep CORE-007/008 closed until BENCH-007 mechanism maps and direct prior-art controls justify
   them.

The exact executable handoff is in
`ara/prompts/continue-structsplat-actual-rate-research.md`.
