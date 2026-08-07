# Current blockers and external techniques

**Updated:** 2026-07-14. BENCH-007 has now supplied the actual-rate decision and rejected the
current tensor-WSE compression claim at its development gate. The main blocker is no longer
missing plumbing; it is the absence of a promotable mechanism and held-out evidence.

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
- BENCH-007 completed 288 fits and 1,152 latest validated SSPL1 candidates. Tensor-WSE gained at
  0.5 bpp but tied the strongest gradient control at 1.0 bpp, missed the BD-rate magnitude, cost
  47.5% more, and failed the texture guard. Stage 2 is prohibited by the frozen gate.

## Blocking issues

### 1. No positive held-out actual-rate evidence

BENCH-007 now provides exact byte caps, complete streams, original-pixel denominators, independent
equal-search fits, persisted-stream validation, nondominated curves, and no-extrapolation BD-rate.
Its Stage-1 gate failed, so the untouched DIV2K validation/Kodak confirmation was correctly not
run. The infrastructure is decision-grade; the current tensor-WSE method is not promotable.

### 2. The native closest handcrafted baseline is still missing

[Structure-Guided Allocation](https://arxiv.org/abs/2512.24018) already couples SLIC/Sobel
structure classes to allocation, geometry regularization, and adaptive covariance precision. A
random/grid/GaussianImage analogue is no longer the strongest control for the tensor/WSE claim.
BENCH-007 now includes an explicitly local, assumption-frozen SLIC/Sobel transplant plus a stronger
local gradient control. Tensor-WSE beats the SLIC/Sobel transplant, but the gradient arm is the
frozen strongest direct control and defeats promotion. BENCH-005 should still run the official
Structure-Guided method if code becomes available, but that independent native lane cannot rescue
the failed common-renderer claim.

### 3. Count, parameter BPP, and actual rate are mixed across the field

SSPL1 bytes, a 32-byte float payload, Image-GS analytical bits, GaussianImage's
`56N+1728` no-EC formula, checkpoint size, and AIR/native quantized rate answer different
questions. Every report and table must use separate columns, null unavailable quantities, and
forbid substitutions. Header, ranges, codebooks, masks, base layers, and side information count.

### 4. Common controls and native methods answer different questions

Paper-inspired rows inside StructSplat isolate mechanisms under one renderer/fitter. Native
executions preserve external validity but confound initialization, representation, loss,
optimizer, schedule, and code. BENCH-007 did not identify a promotable renderer/objective
interaction, so BENCH-008 is not authorized. Native-authentic results remain a separate lane.

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
codec does not derive geometry at the decoder. BENCH-007 did not establish a surviving structural
mechanism or layout bytes as the binding loss, so COMP-005 is not authorized. Decoder-synchronized
geometry may reappear only under a materially new question and evidence program.

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
- **Actual-rate decision plumbing:** BENCH-007 completed exact-cap RDO, equal search, cold-stream
  validation, statistics, conventional context, and F5--F9. The result is negative, not missing.
- **Common-renderer direct controls:** local SLIC/Sobel, gradient, uniform-WSE, and random controls
  are complete. The native official-method lane remains separate.

## Recommended execution order

1. Freeze the BENCH-007 negative result. Do not tune its eight images or run Stage 2 as rescue.
2. Close the exact tensor-WSE compression claim and keep BENCH-008/COMP-005 unauthorized.
3. If research continues, run a new ideation/prior-art pass around a materially different
   hypothesis, null, compute/texture guards, and disjoint development set before implementation.
4. Continue native actual-RD coverage only as an independent BENCH-005 benchmark/external-validity
   lane.
5. Keep CORE-007/008 design-only until a new question and direct Contour-Aware 2DGS/WIPES controls
   justify them.

The exact executable handoff is in
`ara/prompts/continue-structsplat-actual-rate-research.md`.

A composition-focused pipeline design and ordered evidence program consistent with this order —
speed, convergence, quality, and complete-stream rate, built only from already-measured mechanisms
and existing task gates — is recorded in
[research/2026-08-07-fast-convergent-compressive-pipeline-design.md](research/2026-08-07-fast-convergent-compressive-pipeline-design.md)
(2026-08-07, proposal only; no claim or default change).
