# Claims

## C01: StructSplat is primarily a controllable stage-search harness

StructSplat's current strength is not a final codec claim; it is that initialization, renderer,
fitting, densification, pyramid, and codec choices are exposed as testable stages with JSON/CSV
evidence. See `benchmarks/stage_search.py`, ADR-0010, and ABL-002.

## C02: Feature-aware scale caps are searchable controls, not a current default candidate

Feature-adaptive caps reduced scale outliers and improved the small COCO screens, but the
fair-density difficult-four protocol rejected both the old resolution-scaled cap and the INIT-008
feature-relative repair as defaults. `feature_rel` improved strongly over the old absolute cap, but
still averaged -0.3733 dB PSNR versus matching uncapped rows over 48 paired cells and lost badly at
budget 2000. Keep scale caps as stage-search controls; do not present them as a default candidate
without a new task and stronger evidence. See ADR-0012 and
`ara/evidence/init008-feature-relative-scale-caps-2026-07-07/`.

## C03: Exact CUDA is a semantic accelerator, while gsplat is a comparator

`renderer=cuda` and `renderer=cuda_additive` implement StructSplat's own equations and can be used
for speed without changing the method. `renderer=gsplat` remains useful for comparison, but its
alpha/sum semantics are not equivalent to the normalized reference. See ADR-0011 and trace node N15.

## C04: Current cross-repo evidence is metric-split

The cross-repo matrices are matched executable policy analogues under StructSplat's harness, not
native external repository benchmarks. The 2026-07-03 COCO matrix showed the best-searched
StructSplat policy row leading PSNR while GaussianImage/Instant-GI-style analogue rows remained
competitive or better on some MS-SSIM/LPIPS slices. A 2026-07-04 held-out Kodak4 matrix added a
`structsplat_shipped_defaults` row; current and shipped-default StructSplat rows both beat the
GaussianImage/Image-GS analogue rows at max-side 160 / 80 iterations, while Instant-GI was not run
without its external script. Stronger claims still need the full ABL-001/ABL-004 sweep, native
external-repo runs where possible, and a declared target metric. See trace node N17 and
`ara/evidence/abl004-kodak4-cross-repo-2026-07-04/`.

## C05: The flanking hypothesis is retired; structured placement is the claim that survives

Across the 2026-07-04 8-image exact-CUDA screen
(`ara/evidence/abl004-stage-screen-8img-cuda-2026-07-04/`), the 2026-07-05 fair density-control
comparison (`ara/evidence/fair-density-control-difficult4-2026-07-05/`), and the completed ABL-006
successive-halving confirmation (`ara/evidence/abl006-complete-2026-07-07/`),
`aniso_flanking` never leads a decision-grade slice and is eliminated at stage 1. ABL-006's
3-seed finalist confirmation gives a budget-specific answer: `aniso_onedge` has the higher
2000-Gaussian mean PSNR but not a significant paired lead over `quadtree_wse`; `quadtree_wse` is
the clear 5000-Gaussian PSNR winner (+0.0930 dB, 95% CI [+0.0168, +0.1700]); and `quadtree_wse`
has a small non-significant 10000-Gaussian PSNR lead while `aniso_onedge` has higher MS-SSIM.
The claim that survives is structure-tensor-driven, density-aware placement, with `quadtree_wse`
as the high-budget PSNR default candidate and `aniso_onedge` as the low-budget/MS-SSIM alternative.
Default retirement is INIT-007 (ADR-0013).

## C06: Structure-tensor loss weighting is searchable, not a default

FIT-012 added `loss_weight=tensor`, which weights only the pixel-loss term by normalized
structure-tensor energy while leaving SSIM and all reported metrics unweighted. The fair-regime
difficult-four slice at budgets 2000/5000 produced 16 paired tensor-vs-none cells: mean PSNR was
effectively neutral (+0.0061 dB), edge MAE improved in 10/16 pairs, but mean AUC fell by -0.0107
and the result split by strategy. Tensor weighting helped `aniso_onedge` (+0.2661 dB mean PSNR)
and hurt `quadtree_wse` (-0.2538 dB mean PSNR). Keep it available as a stage-search axis and
default it off unless a future task proves a narrower promoted recipe. See
`ara/evidence/fit012-edge-weighted-loss-2026-07-07/`.

## C07: Pyramid fitting has a local quality candidate, not a shipped default yet

HIER-003 overturned the stale claim that the image pyramid simply loses final quality, but exposed
an AUC loss for the old 750/750 two-level schedule. HIER-004 repaired that local tradeoff with
explicit per-level iteration schedules: `level_iters=[150, 1350]` on the 0.35/0.65 budget split
beats the 750/750 pyramid control by +0.0601 dB mean PSNR and is AUC-neutral versus single-stage
(+0.0011 mean AUC) on the difficult-four 2k/5k slice. HIER-001's additive-renderer follow-up did
not improve the story: on the 512-Gaussian Kodak4 slice, `cuda_additive pyramid` lost to
`cuda_additive single` by -0.3743 dB and to `cuda pyramid` by -0.6075 dB. Keep `pyramid=single`
as the shipped default until larger multi-seed confirmation; use normalized-renderer 150/1350 as
the pyramid quality candidate. See `ara/evidence/hier003-pyramid-diagnosis-2026-07-07/`,
`ara/evidence/hier004-pyramid-convergence-repair-2026-07-07/`, and
`ara/evidence/hier001-additive-pyramid-2026-07-07/`.

## C08: The counted background layer is a low-budget candidate, not a default

CORE-009 added a stage-searchable frozen-geometry Gaussian background layer whose rows count
against `num_gaussians` and whose colors remain learnable. On the difficult-four fair-regime slice
at budgets 1000/2000/5000, `background=frac0.05_grid8` averaged +1.0152 dB PSNR and +0.01412
MS-SSIM over 24 paired cells versus `background=off`, winning PSNR in 22/24 pairs. The effect is
budget-dependent: +1.8768 dB at 1000 rows, +1.1183 dB at 2000 rows, and only +0.0504 dB at 5000
rows, where AUC lost in every pair. Keep the layer searchable and default off. Do not claim a true
additive background/compositing layer until a new ADR and larger confirmation justify changing
renderer semantics. See `ara/evidence/core009-background-layer-2026-07-07/`.

## C09: Feed-forward warm starts need tensor priors and are not default-ready

FF-001 established the learned warm-start path, but the current tiny CNN is not better than the
hand tensor-prior initializer. On the 2026-07-07 held-out Kodak slice, the image-only learned
checkpoint reached only 21.0514 dB (`kodim19`, 512 Gaussians, 200 refinement iterations), while the
image+structure-tensor checkpoint beat random scratch on final PSNR (23.4686 vs 22.9056 dB) and
reached 22 dB faster (69 iterations / 0.128815 s vs 108 iterations / 0.166847 s). The same
tensor-prior checkpoint still lost clearly to `quadtree_wse` (25.3249 dB, AUC 24.1797). Keep
`strategy=feedforward` as an experimental warm-start path; do not promote learned initialization
without a larger architecture or predict-optimize-distill evidence. See
`ara/evidence/ff001-multimage-tensor-ablation-2026-07-07/`.

## C10: Fit-time QAT is searchable, not a codec default

COMP-004 added fit-time `qat_mode`, `lambda_rate`, a differentiable rate proxy, and RD benchmark
lambda sweeps. On the 2026-07-07 Kodak4 slice at 512 Gaussians and four codec ladders, fit-time
STE QAT improves encoded low/mid-bit PSNR over direct post-hoc quantization, but it does not beat
the existing post-fit QAT control overall. Post-fit QAT averaged +0.6580 dB over direct encode,
while the best fit-time lambda averaged +0.6539 dB and barely changed actual bpp. Keep
`qat_mode` / `lambda_rate` searchable for future codec experiments; do not promote them as the
default compression path yet. See `ara/evidence/comp004-lambda-sweep-2026-07-07/`.

## C11: Renderer memory and support-continuity fixes are opt-in controls

CORE-005 added the compact-support fade equation and opt-in reference-renderer checkpointing.
Support fade is not a default candidate from current evidence: the fair-density difficult-four
slice improved AUC but lost final PSNR overall (-0.1389 dB). Reference render checkpointing is a
memory control, not a quality change; on a 256x256 CUDA reference-render smoke with 3000 Gaussians,
it reduced peak allocated memory delta from 203.55 MB to 29.65 MB at identical loss. Keep
`support_fade=False` and `render_checkpoint=False` by default; enable checkpointing when the Python
reference renderer is memory-bound. See `ara/evidence/core005-render-checkpoint-2026-07-07/`.

## C12: The pinned default is balanced under the review proxy, not universally dominant

- **Statement**: On N73's fixed four-COCO-image, cap-640, max-side-160, 500-step, two-seed fair
  proxy, `structsplat_best_default` remains the best balanced equal-budget default under the
  predeclared PSNR/MS-SSIM/AUC/fit/total-time gate, but it does not strictly dominate every
  analogue row and the artifact cannot support native external, codec, or paper-grade convergence
  superiority claims.
- **Status**: supported
- **Provenance**: user
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: An equal-budget candidate jointly beats the pinned row on all five
  complete-pair core metrics under the familywise audit, or a broader native comparison changes
  the stated scope/ranking.
- **Proof**: [`ara/evidence/fair-gaussian-variants-full-external-same-hparams-2026-07-10/default_dominance.csv`, `ara/evidence/fair-gaussian-variants-full-external-same-hparams-2026-07-10/summary.md`]
- **Dependencies**: [C04]
- **Tags**: default-selection, proxy, dominance, claim-boundary
- **From staging**: O31

## C13: Native GaussianImage++ is a short-horizon time-quality tradeoff

- **Statement**: In the BENCH-005 matched-axis slice (COCO4 x seeds 0/1, max-side 160, cap 640,
  500 requested steps), native GaussianImage++ is faster than StructSplat default by 0.4284 s mean
  fit time, but its native-minus-StructSplat gains are -5.0678 dB PSNR, -0.05142 proxy MS-SSIM,
  -0.1886 LPIPS, and -7.1638 AUC; this establishes a proxy time-quality tradeoff, not a global
  method ranking.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A provenance-valid rerun on the same cells reverses one or more
  stated paired means, or broader native-authentic/multi-budget evidence invalidates extending the
  bounded tradeoff beyond this slice.
- **Proof**: [`ara/evidence/bench005-native-gi-plus-proxy-2026-07-10/paired_native_vs_structsplat_summary.csv`, `ara/evidence/bench005-native-gi-plus-proxy-2026-07-10/summary.md`]
- **Dependencies**: [C04, C12]
- **Tags**: native-reference, GaussianImage++, convergence, performance, proxy
- **From staging**: O32

## C14: Geometry-consistent regularization is a quality candidate, not the default

- **Statement**: Dense target-gradient-weighted Sobel regularization at weight 0.015 improves
  StructSplat quality/convergence means in the current COCO proxy and Kodak4 confirmation, but its
  larger-resolution time cost blocks balanced-default promotion; every-two/every-four cadence does
  not reliably preserve the quality benefit.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A randomized multi-seed/multi-density confirmation shows no dense
  quality/AUC advantage, or a synchronized implementation removes the cost and passes all
  familywise promotion gates.
- **Proof**: [`ara/evidence/fit013-geometry-consistency-2026-07-10/proxy/default_dominance.csv`, `ara/evidence/fit013-geometry-consistency-2026-07-10/kodak4/default_dominance.csv`, `ara/evidence/fit013-geometry-consistency-2026-07-10/schedule/default_dominance.csv`]
- **Dependencies**: [C06, C12]
- **Tags**: geometry-loss, Sobel, convergence, perceptual-quality, default-selection
- **From staging**: O33

## C15: Native Image-GS fixed-N loses final quality at the short proxy horizon

- **Statement**: In the hash-verified BENCH-005 fixed-N lane (COCO4 x seeds 0/1, max-side 160,
  cap/start 640, 500 steps), native Image-GS minus StructSplat gains are -3.6011 dB PSNR,
  -0.01879 proxy MS-SSIM, and -0.1842 LPIPS; the familywise PSNR/proxy-MS relation supports
  StructSplat on this ablation, but does not establish strict implementation dominance.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A provenance-valid rerun on the same decoded pixels reverses the
  paired final-quality relation, or evidence is presented as strict speed/AUC/global dominance
  without resolving the recorded initialization, timing, and trajectory-protocol differences.
- **Proof**: [`ara/evidence/bench005-native-image-gs-2026-07-10/fixed_paired_summary.csv`, `ara/evidence/bench005-native-image-gs-2026-07-10/fixed_summary.md`]
- **Dependencies**: [C12]
- **Tags**: native-reference, Image-GS, fixed-N, proxy, final-quality
- **From staging**: O35

## C16: Image-GS is a metric tradeoff at the 5k algorithm-profile horizon

- **Statement**: In the one-seed COCO4 max-side-160 cap-640/start-320 5,000-step lane, native
  Image-GS minus StructSplat gains are -0.3840 dB PSNR (95% CI [-2.3698,+1.1997]), +0.01608
  proxy MS-SSIM [+0.00074,+0.03142], and -0.0443 LPIPS gain [-0.0652,-0.0243]; this is a
  heterogeneous tradeoff, not a global winner.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A hash-verified multi-seed rerun at the same algorithm profiles
  yields same-direction gains across PSNR, proxy MS-SSIM, and LPIPS, or official-environment/
  full-resolution evidence overturns the stated bounded tradeoff.
- **Proof**: [`ara/evidence/bench005-native-image-gs-2026-07-10/siggraph25_paired_summary.csv`, `ara/evidence/bench005-native-image-gs-2026-07-10/siggraph25_summary.md`]
- **Dependencies**: [C12, C15]
- **Tags**: native-reference, Image-GS, convergence-horizon, perceptual-quality, proxy
- **From staging**: O36

## C17: Persistent generation-density covariance filtering does not transfer as a default

- **Statement**: For the current normalized StructSplat best-default recipe on COCO4 x seeds
  0/1 at cap 640/500 steps, persistent birth-cohort covariance filters with alpha 9*pi, 18*pi,
  and 36*pi all lose PSNR, proxy MS-SSIM, AUC, and LPIPS versus filter-off; the exact mechanism
  is therefore not default-worthy.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A paired confirmation finds a positive filter strength that beats
  filter-off on final PSNR, proxy MS-SSIM, AUC, and LPIPS without extra capacity, or a different
  compositor invalidates applying this bounded conclusion to that renderer.
- **Proof**: [`ara/evidence/fit014-generation-caf-2026-07-10/default_dominance.csv`, `ara/evidence/fit014-generation-caf-2026-07-10/summary.md`]
- **Dependencies**: [C12]
- **Tags**: covariance-filter, GaussianImage++, normalized-compositor, dead-end, default-selection
- **From staging**: O37

## C18: Same-final-count checkpoint selection repairs long-horizon terminal regression

- **Statement**: On COCO4 x seeds 0/1 at N=640, same-final-count best-PSNR checkpoint selection
  improves the pinned recipe's 5,000-step endpoint over its own terminal state by +0.7702 dB
  PSNR, +0.00892 MS-SSIM, and +0.0076 LPIPS gain across eight runs, while the corresponding
  500-step audit is effectively neutral. This supports an opt-in long-horizon quality policy,
  not replacement of the general terminal default.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: Broader matched-budget/resolution evidence shows no long-horizon
  endpoint gain, selected and terminal states do not have identical final counts, or the
  short-horizon/perceptual tradeoffs become large enough to invalidate opt-in use.
- **Proof**: [`ara/evidence/fit015-full-count-checkpoint-2026-07-10/run.md`, `ara/evidence/fit015-full-count-checkpoint-2026-07-10/long_checkpoint_selection.csv`, `ara/evidence/fit015-full-count-checkpoint-2026-07-10/short_checkpoint_selection.csv`]
- **Dependencies**: [C12]
- **Tags**: checkpoint-selection, long-horizon, terminal-regression, default-selection
- **From staging**: O39

## C19: Official Image-GS evidence remains horizon- and metric-dependent

- **Statement**: In the verified official Image-GS environment on COCO4 at max-side 160 and
  N=640, StructSplat wins the bounded two-seed 500-step final-quality relation, whereas at
  5,000 steps the StructSplat checkpoint candidate is +0.3601 dB in PSNR and better in LPIPS
  while Image-GS is +0.01038 in proxy MS-SSIM. These lanes establish a horizon- and
  metric-dependent tradeoff, not strict implementation or global dominance.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A provenance-valid rerun of the same official-environment cells
  reverses the stated paired means, or broader native-resolution/multi-budget evidence supports
  same-direction gains across all final-quality metrics under a shared protocol.
- **Proof**: [`ara/evidence/bench005-official-native-references-2026-07-10/run.md`, `ara/evidence/bench005-official-native-references-2026-07-10/image_gs_500_vs_default.csv`, `ara/evidence/bench005-official-native-references-2026-07-10/image_gs_5000_vs_checkpoint.csv`]
- **Dependencies**: [C15, C16, C18]
- **Tags**: native-reference, Image-GS, official-environment, convergence-horizon, metric-tradeoff
- **From staging**: O42

## C20: Native GaussianImage needs explicit horizons and is a 5k tradeoff

- **Statement**: In the official base-GaussianImage fixed-N COCO4 lanes, 500 steps leaves the
  native optimizer far from competitive final quality despite faster fitting. At 5,000 steps,
  GaussianImage is 6.4448 s faster and +0.01298 in proxy MS-SSIM, while the StructSplat
  checkpoint candidate is +0.1207 dB in PSNR, better by 0.0253 LPIPS gain, and +1.5337 in
  diagnostic AUC. This is a horizon-specific tradeoff, not a global ranking.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A provenance-valid same-cell rerun reverses the stated paired
  means, or shared native-resolution/multi-budget timing and trajectory evidence produces a
  consistent winner across quality, convergence, and performance.
- **Proof**: [`ara/evidence/bench005-official-native-references-2026-07-10/run.md`, `ara/evidence/bench005-official-native-references-2026-07-10/gaussianimage_500_vs_default.csv`, `ara/evidence/bench005-official-native-references-2026-07-10/gaussianimage_5000_vs_checkpoint.csv`]
- **Dependencies**: [C13, C18, C19]
- **Tags**: native-reference, GaussianImage, official-environment, convergence-horizon, metric-tradeoff
- **From staging**: O43

## C21: The 2x low-pass loss-target warmup fails the checkpoint-controlled guard

- **Statement**: On COCO4 x seeds 0/1 at max-side 160, cap/start 640/320, and 500 steps, the
  FIT-016 2x low-pass-to-full loss curriculum loses 0.1645 dB selected PSNR (95% image-bootstrap
  CI [-0.2856,-0.0677]), 0.00068 MS-SSIM, and 0.0716 AUC versus the otherwise identical
  same-final-count checkpoint control. It is not default-worthy and does not pass the
  preregistered guard for a 5k confirmation.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A provenance-valid rerun of the same exact-treatment cells reverses
  the selected-PSNR interval and quality/convergence direction, or a materially different
  multiscale mechanism is incorrectly treated as evidence for this exact warmup.
- **Proof**: [`ara/evidence/fit016-lowpass-curriculum-2026-07-10/run.md`, `ara/evidence/fit016-lowpass-curriculum-2026-07-10/lowpass_vs_checkpoint_summary.csv`]
- **Dependencies**: [C12, C18]
- **Tags**: loss-curriculum, low-pass, convergence, dead-end, default-selection
- **From staging**: O44

## C22: Checkpoint selection is density-dependent long-fit protection

- **Statement**: Across 72 Kodak4 same-trajectory audits spanning max-side {160,240,320},
  N={1280,2560,5120}, seeds 0/1, and 5,000 steps, same-final-count checkpoint selection gains
  +0.4884 dB pooled PSNR (95% image-bootstrap CI [+0.4167,+0.5304]), +0.00433 MS-SSIM, and
  +0.00736 LPIPS. The PSNR gain falls from +1.0380 dB at N=1280 to +0.0458 dB at N=5120, so
  the policy is a confirmed sparse/moderate-density long-fit option but not the universal default.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A provenance-valid multi-image rerun removes the positive pooled
  interval or density trend, selected/terminal endpoints violate equal-final-count semantics, or
  broader evidence supports the preregistered >=+0.10 dB gain in every resolution/count stratum.
- **Proof**: [`ara/evidence/fit015-checkpoint-broad-2026-07-10/run.md`, `ara/evidence/fit015-checkpoint-broad-2026-07-10/checkpoint_selection_summary.csv`, `ara/evidence/fit015-checkpoint-broad-2026-07-10/checkpoint_selection_matrix.csv`]
- **Dependencies**: [C18]
- **Tags**: checkpoint-selection, long-horizon, density, terminal-regression, default-selection
- **From staging**: O45

## C23: GaussianImage no-EC rate is analytical and depends on live decoder state

- **Statement**: The released GaussianImage Cholesky Kodak path uses native-orientation images,
  N={800,1000,3000,5000,7000,9000}, seed 1, 50k representation steps, and 50k QAT steps. Its
  fixed-width no-entropy representation is `56N+1728` bits, but `compress_wo_ec()` omits live
  quantizer decoder state; therefore this lane has analytical rate only and must report null
  actual serialized bytes/bpp.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The hash-pinned official source is shown to serialize every required
  codebook/scale/offset and payload byte, or a later upstream release adds a self-contained stream
  whose exact bytes are measured and explicitly distinguished from this audited commit.
- **Proof**: [`ara/evidence/bench005-gaussianimage-release-rd-audit-2026-07-10/run.md`, `tasks/BENCH-005-native-reference-pipelines.md`, `benchmarks/README.md`]
- **Dependencies**: [C20]
- **Tags**: native-reference, GaussianImage, QAT, analytical-rate, codec-boundary
- **From staging**: O46
