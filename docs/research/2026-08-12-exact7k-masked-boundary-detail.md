# Exact-7k masked boundary and thin-detail research

## Question

Can the max-side-1200 Janelle C0001 foreground be reconstructed without coverage holes and with
better hair/fine detail at exactly 7,000 Gaussians, while every centre and compact support remains
inside the mask? Does spatially equal error make a later count increase reliably useful?

This is an exposed single-image development study. It deliberately defers the approximately
57,600-row density-equivalent and native-camera experiments.

## Why equal raw error is not the target

For a fixed row budget, an efficient allocation equalizes the *marginal reduction in the chosen
loss from the next row*, subject to topology, containment, and perceptual constraints. It does not
equalize raw pixel error. Hair, lace, boundaries, and texture can retain higher residual error
because their local approximation problem is harder. Conversely, a smooth region can have low
error but still be over-allocated if its next row has almost no value.

This distinction matters for scaling. More ordinary rows can reduce approximation error only where
the representation is feasible and the allocator can reach the residual. They cannot repair a
mask component with no legal centre under an unchanged minimum scale and containment margin.
Topology/representability must therefore be established before marginal allocation is interpreted.

HIER-031 reports tile residual SSE, local complexity, row density, and a common next-row gain proxy.
The proxy coefficient of variation is `0.7922` for the HIER-030 control and `0.8216` for the selected
endpoint, so the experiment does not show marginal-gain equalization. It shows that a hard
topology reserve can remove starvation while preserving useful appearance capacity.

## Geometry feasibility

The resized mask contains 87,639 active pixels and four connected components. With margin 0.75,
three-sigma compact support, and the ordinary 0.35-pixel scale floor:

- a legal ordinary centre requires SDF at least `0.75 + 3 * 0.35 = 1.80` pixels;
- 980 mask pixels are outside isotropic reach from such centres (an upper bound because a certified
  tangent ellipse can reach some of them);
- three connected components containing ten pixels have no legal ordinary centre at all, which is
  a count-independent lower bound;
- a 0.08-pixel micro row has certificate radius 0.99 pixels and can be centred at every active
  pixel in this mask.

The current scale floor therefore makes “just add more ordinary Gaussians” mathematically
insufficient for exact topology. The needed representation is a fixed certified micro cohort plus
ordinary appearance rows.

This is consistent with the motivation—not a claimed reimplementation—of skeleton-aware thin-
structure allocation in [Prior-Enhanced Gaussian Splatting](https://arxiv.org/abs/2512.11356) and
topology-sensitive skeleton overlap in
[clDice](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html).

## Frozen sequential study

All scored endpoints contain exactly 7,000 rows and only means, log-scales, rotations, and RGB
coefficients. Every endpoint passes exact centre/support/reconstruction containment. The protocol
and every sequential amendment are recorded before execution in HIER-031.

The initial one-micro-per-hole exchange closes all 869 control holes but spends appearance rows and
loses 5.15 dB of interior PSNR. Function-preserving envelope merges fund the same micro reserve
more gently, but the zero-hole endpoint still loses 2.39 dB inside. Geometry recovery around the
fixed reserve restores interior/detail quality but moving long ordinary supports opens 221 boundary
sites. A terminal closure removes those holes but again spends too much interior capacity.

The final non-tuned recovery freezes both the certified micro cohort and every ordinary row whose
centre is not deeper than 6.75 pixels. Deep-only fitting improves fine-detail and interior metrics
but opens 37 sites reached by long supports from deep centres. Applying the unchanged terminal
closure funds 37 more micro rows and is the first endpoint to pass the original topology/interior
guard. No further method development is permitted in this task.

## Results

| Endpoint | Raw holes | weak coverage `<0.05` | PSNR | boundary PSNR | interior PSNR | high-pass MSE | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| HIER-030 cold additive control | 869 | 1,649 | 21.5745 | 12.4807 | 35.3131 | 0.000113735 | 0.13494 |
| selected deep-only + terminal closure | **0** | **743** | 23.8589 | 14.8367 | 36.0677 | 0.000106562 | 0.12085 |
| untouched fixed-capacity pipeline | 933 | 1,221 | **25.2175** | **16.0939** | **39.8058** | **0.000088250** | **0.07828** |
| pipeline with capacity-time boundary recycle | 955 | 1,278 | 25.1792 | 16.0572 | 39.7181 | 0.000088776 | 0.07935 |

The selected endpoint uses 910 micro rows and 6,090 ordinary rows. Against the frozen control it
gains 2.2844 dB overall, 2.3560 dB at the boundary, and 0.7546 dB in the interior; deep high-pass
MSE falls 6.31%, Sobel MSE falls 11.40%, and LPIPS falls 10.45%. Laplacian MSE worsens 4.56%, so
fine-detail evidence is mixed rather than uniformly positive. Weakly covered pixels fall 54.9%,
all raw and thin-ridge holes disappear, and support/reconstruction outside remain exactly zero.

The ordinary pipelines are clearly sharper but violate the no-hole requirement. Visual review
shows broken hair strands and a broad zero-support fringe in their hole maps. The selected endpoint
keeps the strands connected but remains visibly soft and has larger additive coefficients
(`|c|max=15.998`, q99=6.266), which is a conditioning warning.

## Scientific disposition

The idea is not stupid, but it needs two corrections:

1. Equalize marginal value, not raw error.
2. Make every target structure representable before expecting scaling to help.

At 7,000 rows, the two-cohort method gives a topology-safe and measurably better endpoint than the
HIER-030 control. It does not match the full pipeline's appearance, prove generalization, or show
that later 57.6k scaling will improve every region automatically. A later scaling experiment
should preserve a certified topology reserve, allocate new ordinary rows by measured marginal
gain, and compare against the same-count untouched pipeline on disjoint images. It should report
coverage and topology as constraints rather than allowing higher average PSNR to hide holes.

## Evidence and limits

- Portable report:
  `results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_2026-08-12/index.html`
- Evidence note: `ara/evidence/hier031-exact7k-masked-boundary-detail-2026-08-12/run.md`
- Report status: dirty-source, exposed C0001, seed 0, RTX 3050, self-reviewed diagnostic;
  `formal_claim_ready=false`.
- Missing: disjoint images, multiple seeds/devices, native 5328x4608, density parity, rate/byte
  matching, held-out mask-topology tests, and independent scientific/visual review.
