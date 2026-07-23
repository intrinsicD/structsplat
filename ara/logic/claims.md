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

## C24: Isotropic signed-Gaussian sampled-add scoring fails the recovery guard

- **Statement**: On FIT-017's shared-start COCO4 x seeds 0/1 guard at max-side 64 and
  N=64->80, `signed_gaussian` improves immediate PSNR by +0.5199 dB versus `legacy_abs` but
  changes post-20/post-100 PSNR by -0.0318/-0.2301 dB and wins only 3/8 post-20 pairs. This exact
  isotropic coherent-error score is refuted under the preregistered recovery criteria.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: A source-bound rerun of the same shared-start cells passes the
  +0.10 dB post-20, 75% sign-agreement, and -0.05 dB post-100 gates; materially anisotropic,
  denominator-aware, or differently colored interventions do not count as this exact score.
- **Proof**: [`tasks/FIT-017-matched-residual-densification.md`, `benchmarks/sampled_add_score_compare.py`, `a68337d`]
- **Dependencies**: []
- **Tags**: densification, matched-filter, signed-residual, dead-end, normalized-renderer
- **From staging**: O49

## C25: Progressive WSE ordering repairs uniform low-count prefixes without changing the set

- **Statement**: On INIT-009's clean-commit uniform Euclidean audit with eight independently
  generated M=2048->N=256 terminal sets, Yuksel-style progressive ordering preserves every
  terminal set and improves both normalized minimum spacing and inverse normalized coverage-hole
  over candidate-index ordering in all 32 descriptive seed/prefix pairs at counts 16/32/64/128;
  the ordering subroutine takes 14.2% of terminal selection time in this microbenchmark.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: A source-bound rerun changes terminal membership, fails either mean
  metric at a preregistered prefix, falls below 75% joint paired wins, or exceeds the 25%
  subroutine-overhead gate. Anisotropic/quadtree/image-quality claims require separate evidence.
- **Proof**: [`ara/evidence/init009-wse-prefix-audit-2026-07-13/aggregate.json`, `ara/evidence/init009-wse-prefix-audit-2026-07-13/prefix_rows.csv`, `ara/evidence/init009-wse-prefix-audit-2026-07-13/config.json`]
- **Dependencies**: []
- **Tags**: weighted-sample-elimination, progressive-order, prefix, blue-noise, correctness
- **From staging**: O50

## C26: The completed 168 KiB lane is high-rate policy evidence, not compression evidence

- **Statement**: The external-present BENCH-006 execution completed 320/320 common-harness cells
  (40 methods × four COCO training images × two seeds), but its fixed 172,032-byte float payload is
  71.68–81.15 bpp at the prepared resolutions. Its actual SSPL1 streams are about 22 bpp, while the
  prepared lossless target PNGs average about 17.99 bpp. The report supports bounded optimizer and
  policy comparisons under a normalized StructSplat harness; it cannot establish compression SOTA
  or native superiority over paper methods.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: Recalculation from the committed source dimensions and complete
  streams contradicts the stated rates, or the report is incorrectly relabeled despite changing
  neither its data, native-method execution, nor rate protocol.
- **Proof**: [`ara/evidence/bench001-external-complete-2026-07-13/run.md`,
  `ara/evidence/storage-budget-168k-sota-audit-2026-07-13.md`,
  `results/storage_budget_168k_external_present/image_storage.csv`,
  `results/storage_budget_168k_external_present/metrics.csv`]
- **Dependencies**: [C04, C12]
- **Tags**: actual-rate, fixed-storage, claim-boundary, native-reference, compression
- **From staging**: direct 2026-07-13 repository/SOTA audit

## C27: StructSplat's broad structure-aware novelty boundary is occupied

- **Statement**: Broad claims for structure-aware 2D Gaussian allocation/orientation/precision,
  normalized local ownership, progressive Gaussian coding, segmentation-gated boundaries, learned
  initialization, and generic clustered quantization are directly covered or threatened by
  Structure-Guided Allocation, Image-GS/SAD, P-GSVC, Contour-Aware 2DGS, Instant-GI and later
  learned samplers, and CGVQ. The narrower training-free tensor-metric/WSE actual-rate claim has
  now failed its preregistered development gate; any future positive claim must use a materially
  different question and disjoint evidence program.
- **Status**: supported as a prior-art boundary; the current narrow positive claim is rejected
- **Provenance**: ai-suggested
- **Crystallized via**: prior-art-audit
- **Falsification criteria**: A primary-source audit shows that one named threat lacks the attributed
  mechanism, or a more specific StructSplat claim is demonstrated to predate or be irreducible to
  those mechanisms, or independent evidence establishes a materially different StructSplat
  mechanism. Search cannot prove global novelty, and BENCH-007's failed current formulation does
  not rule out every possible future mechanism.
- **Proof**: [`ara/evidence/storage-budget-168k-sota-audit-2026-07-13.md`,
  `ara/evidence/research-portfolio-2026-07-13.md`,
  `ara/evidence/bench007-stage1-killing-pilot-2026-07-14/run.md`,
  `tasks/BENCH-007-actual-rate-structure-phase-diagram.md`]
- **Dependencies**: [C05, C25, C26]
- **Tags**: novelty-boundary, prior-art, structure, actual-rate, research-scope
- **From staging**: 2026-07-13 primary-source research audit

## C28: The current tensor-WSE actual-rate formulation fails its development promotion gate

- **Statement**: On the frozen eight-image DIV2K Stage-1 matrix, tensor-WSE beats the strongest
  local gradient control by `+0.3457 dB` at 0.5 bpp but only `+0.0089 dB` at 1.0 bpp, reaches
  `-4.5417%` mean PSNR BD-rate rather than the required `-10%`, costs `1.4752x` fit plus equal
  search time, and increases texture MSE by `7.2883%` beyond the 5% guard. The preregistered gate
  fails; Stage 2 is not authorized.
- **Status**: supported development-set negative decision; not held-out evidence
- **Provenance**: ai-suggested
- **Crystallized via**: adversarial-experiment-design
- **Falsification criteria**: The source-bound latest-cell analysis is incomplete or incorrect, a
  frozen gate is shown to have been misapplied, or an exact independent rerun of the same manifest
  materially changes the gate outcome. A new formulation or dataset does not falsify this bounded
  result; it answers a new question.
- **Proof**: [`ara/evidence/bench007-stage1-killing-pilot-2026-07-14/run.md`,
  `tasks/BENCH-007-actual-rate-structure-phase-diagram.md`]
- **Dependencies**: [C26, C27]
- **Tags**: actual-rate, tensor-wse, negative-result, development-gate, claim-boundary
- **From staging**: 2026-07-14 BENCH-007 Stage-1 decision

## C29: SAD-style alpha-0.7 responsibility density fails the FIT-018 recovery guard

- **Statement**: On FIT-018's deterministic shared-start COCO4 x seeds 0/1 guard at max-side 64
  and N=64->80, `responsibility_alpha0.7` changes immediate/post-20/post-100 PSNR by
  -0.0623/-0.0198/-0.0411 dB versus the preselected `support` comparator, wins only 4/8 post-20
  pairs, and adds 1.8% to total-100 time. This exact SAD-style transfer is refuted under the
  preregistered +0.10 dB and 6/8-pair recovery gate.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: A source-bound deterministic rerun of the same shared-start cells
  passes the +0.10 dB post-20 and 6/8 sign-agreement gates while preserving post-100 quality,
  count, finiteness, and time. A different alpha, group-level gauge-invariant score, primitive,
  or fixture is a new hypothesis and does not falsify this bounded result.
- **Proof**: [`ara/evidence/fit018-responsibility-guard-2026-07-15/aggregate.json`,
  `ara/evidence/fit018-responsibility-guard-2026-07-15/rows.csv`,
  `ara/evidence/fit018-responsibility-guard-2026-07-15/rerun_aggregate.json`,
  `tasks/FIT-018-responsibility-error-density.md`]
- **Dependencies**: [C24, C27]
- **Tags**: responsibility, normalized-renderer, densification, transfer, dead-end, recovery
- **From staging**: O52

## C30: Top-16 has a near-lossless single-field oracle ceiling

- **Statement**: On one saved 20,000-Gaussian, 640x480 COCO field, an exact clipped-support
  top-16 responsibility oracle reaches 34.57605 dB versus 34.62571 dB full (-0.04966 dB), retains
  mean/p05 responsibility mass 0.996064/0.977087, and exposes ideal reductions of 49.59% of
  numerically positive contributions or 61.50% of rectangle visits. K=8 loses 1.18945 dB. The
  audit evaluates all candidates before selecting winners, so it supports no latency, throughput,
  FLOP, or deployable-kernel claim.
- **Status**: supported single-field mechanism evidence
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: A source-bound rerun of the same field fails the 2e-5 full-oracle
  parity gate or materially changes the recorded K=16 quality/mass/work counts. Multi-field
  failure would defeat generalization but not erase this exact field result; measured acceleration
  requires an implementation that avoids exhaustive candidate evaluation.
- **Proof**: [`ara/evidence/topk-responsibility-oracle-2026-07-15/result.json`,
  `ara/evidence/topk-responsibility-oracle-2026-07-15/rows.csv`,
  `ara/evidence/topk-responsibility-oracle-2026-07-15/config.json`]
- **Dependencies**: []
- **Tags**: responsibility, top-k, oracle, performance-ceiling, single-field, normalized-renderer
- **From staging**: O53

## C31: Exact opacity refinements expose row-allocation non-commutation

- **Statement**: On FIT-019's deterministic eight-family x seeds 0/1 procedural suite, replacing
  every even canonical group by two co-located half-opacity rows preserves normalized renders
  within `8.345e-7`; aggregate-first responsibility scores match canonical scores within
  `2.701e-6` relative error at alpha 0.7 and `2.747e-6` at alpha 1; and quotient top-8 actions
  match all 16 checkpoints at both alphas. Raw alpha-1 physical-group multisets change on both
  seeds for all 8 target families.
- **Status**: supported procedural mechanism evidence
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: A source-bound rerun of the same frozen v2 suite exceeds the `2e-6`
  render or `2e-5` score/child-law tolerance, changes either both-alpha quotient top-8 action, or
  fails the both-seed raw-multiset change gate in at least three target families. Approximate split
  siblings are outside this exact-equivalence claim.
- **Proof**: [`ara/evidence/fit019-opacity-gauge-2026-07-15/algebra.json`,
  `ara/evidence/fit019-opacity-gauge-2026-07-15/aggregate.json`,
  `ara/evidence/fit019-opacity-gauge-2026-07-15/rows.csv`,
  `tasks/FIT-019-opacity-gauge-equivalence.md`]
- **Dependencies**: [C29]
- **Tags**: normalized-renderer, gauge-equivalence, allocation, commutation, responsibility
- **From staging**: O57

## C32: Gauge-quotient allocation fails the FIT-019 recovery-utility guard

- **Statement**: Under FIT-019's fresh-Adam canonical replay, quotient alpha 1 changes post-20
  PSNR by `+0.211079 dB` versus raw gauge-row alpha 1 but wins only 5/8 target families, changes
  post-100 by `-0.600711 dB`, and is `-0.066534 dB` versus canonical support at post-20. It fails
  the preregistered breadth, late-retention, and support-floor gates despite exact commutation,
  equal N=40 counts, finite values, and timing inside the accounting bound.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: A source-bound rerun of the same frozen v2 cells passes all three
  failed quality gates while preserving the commutation, count, finiteness, support, and timing
  conditions. Production optimizer-state continuation, natural images, or approximate grouping
  would answer a new hypothesis rather than erase this bounded restart-recovery result.
- **Proof**: [`ara/evidence/fit019-opacity-gauge-2026-07-15/aggregate.json`,
  `ara/evidence/fit019-opacity-gauge-2026-07-15/rows.csv`,
  `ara/evidence/fit019-opacity-gauge-2026-07-15/replay_aggregate.json`,
  `docs/adr/0014-keep-opacity-gauge-groups-benchmark-only.md`]
- **Dependencies**: [C31]
- **Tags**: gauge-equivalence, quotient-allocation, negative-result, recovery, fresh-optimizer
- **From staging**: O58

## C33: FIT-019 endpoint effects are horizon- and target-sensitive

- **Statement**: In the 16 FIT-019 quotient-minus-raw alpha-1 gauge pairs, the post-20 and
  post-100 PSNR effect changes sign in 7 cells and 3/8 target means. Median effects are
  `+0.0390/-0.0117 dB`; the post-20/post-100 Pearson correlation is `-0.593`, but Spearman is only
  `-0.182`, and excluding the sinusoid reverses both endpoint mean signs. This is descriptive
  evidence that the aggregate endpoint difference is not a stable monotone recovery effect.
- **Status**: supported descriptive post-hoc evidence; FIT-020's one-bend predictor later refuted
  in C34
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: Recomputing the paired effects from the frozen v2 rows changes the
  reported signs, medians, correlations, or leave-one-family-out direction. FIT-020 tests one
  disjoint ranked-deduplication bend and refutes it in C34; a materially different predictor would
  be a new hypothesis.
- **Proof**: [`ara/evidence/fit019-opacity-gauge-2026-07-15/rows.csv`,
  `docs/research/2026-07-15-opacity-gauge-experiment.md`]
- **Dependencies**: [C32]
- **Tags**: convergence, recovery, horizon-reversal, target-heterogeneity, post-hoc
- **From staging**: O60

## C34: FIT-020 response bend fails within-assay held-out prediction and selection

- **Statement**: On FIT-020's frozen six-family within-assay held-out procedural-variant table
  (not confirmation or natural-image evidence), adding the one preregistered early bend to static
  and step-10 controls changes RMSE from `2.961624` to
  `2.964081 dB` (ratio `1.000830`, bootstrap 95% `[0.998582,1.001890]`), leaves sign accuracy at
  25/36, improves only 2/6 family RMSEs, and has `-1.045524 dB` mean prediction-minus-observation
  bias. Early and response models both select C5 for all 12 held-out targets with `1.111568 dB`
  regret, worse than observed step 10 at `0.766896 dB`. The response-prediction and screening gates
  fail despite `SD(y)=3.252924 dB` and 35/36 cells with `|y|>=0.10 dB`.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: A source-bound rerun of the same frozen rows and gates changes the
  compared predictions/actions enough to pass every response-prediction and screening criterion
  while preserving integrity and signal. A different bend, horizon, model, target suite,
  intervention, or optimizer state is a new hypothesis and does not erase this bounded result.
- **Proof**: [`ara/evidence/fit020-response-spectroscopy-2026-07-15/run.md`,
  `ara/evidence/fit020-response-spectroscopy-2026-07-15/aggregate.json`,
  `ara/evidence/fit020-response-spectroscopy-2026-07-15/paired_rows.json`,
  `docs/research/2026-07-15-perturb-recover-spectroscopy.md`]
- **Dependencies**: [C33]
- **Tags**: convergence, recovery, response-bend, held-out, negative-result, selection
- **From staging**: O61

## C35: Ranked deduplication effects are large but family-sensitive in FIT-020

- **Statement**: Under FIT-020's procedural fresh-Adam ranked ticket-deduplication path,
  within-assay held-out procedural-variant step-200 C5/C6/C7 minus C8 means are
  `+1.875776/+2.248652/+1.642426 dB`, but excluding sinusoid
  and chirp gives `-0.280813/+0.429437/+0.408748 dB`. Best step-10 and step-200 arms differ for
  9/12 held-out targets. These are descriptive path effects: site rank, ticket identity, and
  lineage depth co-vary, fixed C6/step-10 were not preregistered promotion claims, and every arm has
  the same N=40 atom schema with no encoded stream.
- **Status**: supported bounded descriptive procedural evidence
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: Recomputing arm/target/family effects from the frozen paired rows
  changes the listed means or 9/12 horizon mismatch. New targets may test transfer, but cannot turn
  the already-exposed fixed-C6 observation into confirmatory evidence.
- **Proof**: [`ara/evidence/fit020-response-spectroscopy-2026-07-15/run.md`,
  `ara/evidence/fit020-response-spectroscopy-2026-07-15/paired_rows.json`,
  `docs/research/2026-07-15-perturb-recover-spectroscopy.md`]
- **Dependencies**: [C34]
- **Tags**: ranked-deduplication, target-heterogeneity, horizon-reversal, post-hoc, procedural
- **From staging**: O62

## C36: Standard birth fails the COMP-006 complete-stream RD gate

- **Statement**: On COMP-006's frozen 18-target procedural development suite at recovery step 20
  and `matched no-edit SSPL1 + 16 bytes`, the oracle best of 16 residual standard-Gaussian births
  changes paired PSNR by `-1.071442 dB` mean and `-0.953260 dB` median versus the strongest
  feasible no-edit, 16-candidate fixed-donor replacement, and 875-mix precision envelope. The
  family-stratified bootstrap 95% interval is `[-1.287276,-0.841740] dB`; all 18 target means and
  all six family means are negative. Exact same-source replay passes, so confirmation is not
  authorized.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: A source-bound rerun of the identical frozen development protocol
  changes the target-level aggregates enough to pass every quality gate while preserving all
  stream, source, environment, and replay checks. A new cap, donor, action bank, QAT horizon,
  target suite, entropy model, atom schema, or codec answers a new hypothesis and does not erase
  this bounded negative result.
- **Proof**: [`ara/evidence/comp006-marginal-cold-stream-rd-2026-07-15/run.md`,
  `ara/evidence/comp006-marginal-cold-stream-rd-2026-07-15/final_summary.json`,
  `docs/research/2026-07-15-marginal-cold-stream-rd.md`,
  `tasks/COMP-006-marginal-cold-stream-rd.md`]
- **Dependencies**: [C28, C35]
- **Tags**: actual-rate, complete-stream, structural-birth, precision-allocation, negative-result,
  development-gate
- **From staging**: O64

## C37: Exact bytes improve control allocation but not structural-birth selection in COMP-006

- **Statement**: In COMP-006's 36 primary +16-byte cells, the complete-stream oracle and nominal-
  raw-bit oracle select the same row in 14/36. Complete-stream selection gains `+0.213081 dB` mean
  PSNR over the proxy selection, with all proxy winners actually feasible, but all 22 disagreements
  concern control allocation: 20 choose a different precision mix and two change replacement to
  precision. Broad branch class agrees in 34/36, and exact rate never changes whether birth wins.
- **Status**: supported bounded descriptive procedural evidence
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: Recomputing the frozen step-20/+16 selections from validated streams
  changes the 14/36 row agreement, +0.213081 dB regret, proxy feasibility, or disagreement
  decomposition. A deployable learned price, new entropy syntax, or sequential edit policy is a
  new claim; this oracle result does not establish one.
- **Proof**: [`ara/evidence/comp006-marginal-cold-stream-rd-2026-07-15/run.md`,
  `ara/evidence/comp006-marginal-cold-stream-rd-2026-07-15/selections.jsonl`,
  `docs/research/2026-07-15-marginal-cold-stream-rd.md`]
- **Dependencies**: [C36]
- **Tags**: actual-rate, proxy-regret, precision-allocation, complete-stream, post-hoc,
  claim-boundary
- **From staging**: O65

## C38: Two exposed m12 exact-cap cells improve all three frozen quality metrics

- **Statement**: In the unsealed same-data COMP-012 mechanics calibration, target-guided SSP2F
  RGB coordinate search matches each strongest SSP2E incumbent's complete bytes and improves all
  three frozen persisted-CUDA metrics on two already-exposed `[12,6,6,8]` development cells.
  Display-rounded conservative three-repeat candidate-minus-incumbent deltas are approximately
  `+0.931405 dB/+0.000196/-0.000449 LPIPS` for Jason at `51,549/51,549` bytes and
  `+0.860618 dB/+0.000171/-0.002742 LPIPS` for Nomao at `52,223/52,223` bytes. These
  nonterminal, different-format cells do not establish
  held-out transfer, a default change, convergence, speed, compression isolation, expressiveness,
  equal decoder complexity, or state-of-the-art improvement.
- **Status**: supported bounded exposed-development calibration
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: Cold-decoding the recorded SSP2E/SSP2F blobs and rescoring them with
  the frozen renderer/metric environment changes any conservative delta's favorable sign or the
  exact complete-byte equality. Independent images, a terminal search, a same-format control, or
  held-out execution would test new transfer and attribution claims rather than erase this bounded
  observation.
- **Proof**: [`ara/evidence/comp012-exposed-exact-cap-calibration-2026-07-17/run.md`,
  `ara/evidence/comp012-exposed-exact-cap-calibration-2026-07-17/visual/metrics.json`,
  `ara/evidence/comp012-exposed-exact-cap-calibration-2026-07-17/streams/jason-baseline.ssp2e`,
  `ara/evidence/comp012-exposed-exact-cap-calibration-2026-07-17/streams/jason-candidate.ssp2f`,
  `ara/evidence/comp012-exposed-exact-cap-calibration-2026-07-17/streams/nomao-baseline.ssp2e`,
  `ara/evidence/comp012-exposed-exact-cap-calibration-2026-07-17/streams/nomao-candidate.ssp2f`,
  `ara/evidence/comp012-exposed-exact-cap-calibration-2026-07-17/index.html`]
- **Dependencies**: []
- **Tags**: actual-rate, equal-byte, RGB-RDO, exposed-development, calibration, claim-boundary
- **From staging**: O84

## C39: BENCH-009 causal recovery is unavailable and cannot rescue the tangent auction

- **Statement**: BENCH-009's source-bound v3 causal audit is unavailable: global calibration is
  `0.268549 < 0.8`, every action-by-horizon causal stratum fails, and independently truncated
  spaces produce negative incremental projector energies. This bounded development result
  authorizes no expressiveness, quality, convergence, performance, compression, or method claim.
- **Status**: supported bounded negative development evidence; causal instrument unavailable
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: Recomputing the identical source-bound v3 ledger makes the causal
  audit available and passes its frozen calibration/projector/control gates. A new instrument,
  action family, or dataset is a new hypothesis.
- **Proof**: [`ara/evidence/bench009-residual-tangent-auction-2026-07-16/run.md`,
  `ara/evidence/bench009-residual-tangent-auction-2026-07-16/causal_audit.json`,
  `tasks/BENCH-009-residual-tangent-auction.md`]
- **Dependencies**: [C36]
- **Tags**: causal-audit, tangent-space, projector, negative-result, development, claim-boundary
- **From staging**: direct BENCH-009 result audit

## C40: Nested affine/carrier extensions fail all BENCH-011 calibration strata

- **Statement**: Corrected BENCH-011 v2 has valid nested algebra, exactly reuses BENCH-009 unit
  identities/base seeds, and reproduces all 96 rows, but zero of four frozen calibration strata
  pass. The spent-data formulation closes without retuning.
- **Status**: refuted spent-data diagnostic
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The same 96 rows retain algebra validity and pass every frozen
  calibration stratum. New radii, actions, targets, or calibration rules are outside this claim.
- **Proof**: [`ara/evidence/bench011-nested-residual-extension-2026-07-16/run.md`,
  `ara/evidence/bench011-nested-residual-extension-2026-07-16/analysis.json`,
  `ara/evidence/bench011-nested-residual-extension-2026-07-16/rows.jsonl`]
- **Dependencies**: [C39]
- **Tags**: nested-subspace, calibration, spent-data, negative-result, claim-boundary
- **From staging**: direct BENCH-011 result audit

## C41: BENCH-012 produces no topology-policy scientific outcome

- **Statement**: BENCH-012 stops at its first source-bound preflight cell because only two of four
  required untruncated work-matched actions are feasible. `scientific_outcomes_scored=false`;
  this is an unavailable feasibility result, not evidence for or against topology-aware policy.
- **Status**: unavailable preflight
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The identical frozen preflight produces all four eligible actions
  without changing the work/truncation rules. A relaxed action set would be a new protocol.
- **Proof**: [`ara/evidence/bench012-topology-policy-preflight-2026-07-16/run.md`,
  `ara/evidence/bench012-topology-policy-preflight-2026-07-16/errors.jsonl`,
  `tasks/BENCH-012-topology-policy-value.md`]
- **Dependencies**: [C40]
- **Tags**: topology, policy-value, preflight, unavailable, claim-boundary
- **From staging**: direct BENCH-012 preflight audit

## C42: The exact local-linear compositor fails BENCH-013 Stage 0

- **Statement**: BENCH-013 v3 passes its analytic affine-reproduction controls and lifecycle
  replay, but compact irregular support produces 82/108 forward failures and 27 permutation
  failures from signed local leverage/ringing; the conditional gradient gate is not reached.
  Stage 1 is prohibited.
- **Status**: refuted procedural mechanism
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The same frozen v3 forward/effective-weight/permutation rows pass
  every registered gate while preserving replay validity.
- **Proof**: [`ara/evidence/bench013-local-linear-compositor-2026-07-16/run.md`,
  `ara/evidence/bench013-local-linear-compositor-2026-07-16/analysis.json`,
  `ara/evidence/bench013-local-linear-compositor-2026-07-16/forward_cells.jsonl`,
  `ara/evidence/bench013-local-linear-compositor-2026-07-16/permutations.jsonl`]
- **Dependencies**: []
- **Tags**: local-linear, affine-reproduction, ringing, stage0, negative-result
- **From staging**: direct BENCH-013 result audit

## C43: The explicit affine carrier fails complete-byte and convergence gates

- **Statement**: BENCH-014's gauge-fixed six-scalar carrier improves correlated synthetic static
  quality/rank with bounded prepared-render cost, but the tail plus residual-color entropy fails
  every complete-byte gate and three bump cells fail the terminal-convergence guard. These
  favorable secondary diagnostics do not rescue the transmitted-tail formulation.
- **Status**: refuted procedural Stage-0 mechanism
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The identical frozen artifact passes every complete-byte and
  convergence gate while preserving its replay and stream identities.
- **Proof**: [`ara/evidence/bench014-affine-carrier-2026-07-16/run.md`,
  `ara/evidence/bench014-affine-carrier-2026-07-16/analysis.json`,
  `ara/evidence/bench014-affine-carrier-2026-07-16/convergence_pairs.jsonl`,
  `ara/evidence/bench014-affine-carrier-2026-07-16/streams.jsonl`]
- **Dependencies**: [C42]
- **Tags**: affine-carrier, complete-stream, convergence, stage0, negative-result
- **From staging**: direct BENCH-014 result audit

## C44: The decoder-synchronized affine lift fails BENCH-015 promotion gates

- **Statement**: BENCH-015 improves equal-stream static quality on smooth procedural families,
  but fails the frozen no-harm, terminal-convergence, and cold-decode gates. Stage 1 and another
  local first-order successor are prohibited.
- **Status**: refuted procedural Stage-0 mechanism
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The same frozen static/convergence/timing rows and exact streams
  pass every failed gate with replay validity intact.
- **Proof**: [`ara/evidence/bench015-decoder-synchronized-lift-2026-07-16/run.md`,
  `ara/evidence/bench015-decoder-synchronized-lift-2026-07-16/analysis.json`,
  `ara/evidence/bench015-decoder-synchronized-lift-2026-07-16/static.jsonl`,
  `ara/evidence/bench015-decoder-synchronized-lift-2026-07-16/timing.jsonl`]
- **Dependencies**: [C43]
- **Tags**: decoder-synchronized, affine-lift, complete-stream, stage0, negative-result
- **From staging**: direct BENCH-015 result audit

## C45: Native SAD reuse fails the frozen BENCH-016 two-rate frontier

- **Statement**: On BENCH-016's valid 144-row, eight-image downsampled-DIV2K development matrix,
  native SAD passes the nominal 0.5-bpp stratum but the 2.0-bpp stratum fails median `+1 dB`,
  worst `-0.25 dB`, and LPIPS-nonworse gates. The joint decision is `abandon SAD reuse`;
  `development_screen_pass=false` and Stage 1/production are unauthorized. SAD is
  recipient-replayable rather than self-contained, so this is not compression evidence.
- **Status**: refuted bounded native-development transfer
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The identical v6 rows pass every two-rate gate and replay check.
  Cross-GPU or new-data results are new evidence. The outcome-responsive v6 integrity repair after
  v4/v5 exposure remains a limitation even though scientific choices were retained.
- **Proof**: [`ara/evidence/bench016-native-sad-frontier-2026-07-16/run.md`,
  `ara/evidence/bench016-native-sad-frontier-2026-07-16/analysis.json`,
  `ara/evidence/bench016-native-sad-frontier-2026-07-16/binding.json`]
- **Dependencies**: [C44]
- **Tags**: SAD, native-baseline, development, two-rate, negative-result, claim-boundary
- **From staging**: direct BENCH-016 result audit

## C46: Log-SPD covariance coding fails the COMP-007 complete-stream gate

- **Statement**: COMP-007 v4 audits 12,096 local complete streams; the log-SPD chart fails seven
  of eight gates, with median whole-container movement `-0.4053%` under zlib and `+0.3426%` under
  zstd versus the required `+1%`. Confirmation remains sealed.
- **Status**: refuted development actual-rate claim
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: A source-bound recomputation of the identical streams passes all
  frozen gates. The committed bundle preserves decision/audit evidence but not the 12,096-stream
  cold-replay corpus, so clean-clone stream replay is not claimed.
- **Proof**: [`ara/evidence/comp007-gauge-free-covariance-2026-07-16/run.md`,
  `ara/evidence/comp007-gauge-free-covariance-2026-07-16/decision_summary.json`,
  `ara/evidence/comp007-gauge-free-covariance-2026-07-16/artifact_audit.json`]
- **Dependencies**: [C41]
- **Tags**: covariance-codec, log-SPD, actual-rate, complete-stream, negative-result
- **From staging**: direct COMP-007 result audit

## C47: Mean-conditioned entropy remains a necessary lower-bound lead

- **Statement**: Both COMP-008 tuples survive the exact optimistic entropy lower-bound gates, so
  `coder_authorized=true` and the decision is `ORACLE_INCONCLUSIVE_IMPLEMENT_CODER`. Because no
  finite coder is measured, this is bounded feasibility evidence and not an actual compression
  improvement; confirmation remains sealed.
- **Status**: supported bounded oracle evidence; actual compression unresolved
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: Exact replay changes the frozen rows/gates/decision, or an
  implementation proves the assumed lower bound infeasible. Success or failure of a real coder
  answers the downstream actual-rate question rather than rewriting this oracle result.
- **Proof**: [`ara/evidence/comp008-sgi-entropy-oracle-2026-07-16/run.md`,
  `ara/evidence/comp008-sgi-entropy-oracle-2026-07-16/analysis.json`,
  `ara/evidence/comp008-sgi-entropy-oracle-2026-07-16/cells.jsonl`,
  `ara/evidence/comp008-sgi-entropy-oracle-2026-07-16/bootstrap_indices_u8.bin`]
- **Dependencies**: [C45, C46]
- **Tags**: entropy-oracle, necessary-condition, compression, inconclusive, claim-boundary
- **From staging**: direct COMP-008 result audit

## C48: Fixed SSP2E v1 fails COMP-009 actual-rate and attribution gates

- **Statement**: COMP-009 actual-rate geometric-mean ratios are `0.986954/0.987294` with 6/8 image
  wins and bootstrap upper ratios `0.999584/0.997035`, while modeled-versus-shuffled aggregate
  ratios `0.9591/0.9616` miss the required 10% attribution effect. Resource gates pass, but both
  tuples fail the conjunctive decision: `ABANDON_FIXED_SSP2E_V1`; confirmation remains sealed.
  COMP-010 independently repairs captured-source lifecycle provenance without strengthening this
  compression result.
- **Status**: refuted development actual-rate claim
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The same frozen actual rows and exact streams pass both tuple-level
  rate and mechanism gates. Lifecycle-only replay changes cannot falsify the scientific result.
- **Proof**: [`ara/evidence/comp009-ssp2e-actual-coder-2026-07-16/run.md`,
  `ara/evidence/comp009-ssp2e-actual-coder-2026-07-16/analysis.json`,
  `ara/evidence/comp009-ssp2e-actual-coder-2026-07-16/actual_rows.jsonl`,
  `ara/evidence/comp009-ssp2e-actual-coder-2026-07-16/streams.sha256`,
  `ara/evidence/comp010-ssp2e-replay-repair-2026-07-16/repair.json`]
- **Dependencies**: [C47]
- **Tags**: SSP2E, actual-rate, spatial-attribution, complete-stream, negative-result
- **From staging**: direct COMP-009/010 result audit

## C49: Backward block reduction remains benchmark-only after full frozen-gate failure

- **Statement**: PORT-004's primary representative RTX-3050 cell reduces exact-backward and
  representative-step time by `57.478%/25.288%`, and an independent run repeats
  `51.284%/24.760%`. However, the confirmation candidate-backward CV is `5.1154% > 5%` and the
  whole-grid direction clause fails. The selector therefore remains benchmark-only.
- **Status**: supported bounded device/source microprofile; promotion refuted
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: An independent rerun of the identical frozen grid passes both
  stability and direction. A new kernel/device or end-to-end fit is separate evidence.
- **Proof**: [`ara/evidence/port004-exact-backward-block-reduction-2026-07-16/run.md`,
  `ara/evidence/port004-exact-backward-block-reduction-2026-07-16/primary/aggregate.json`,
  `ara/evidence/port004-exact-backward-block-reduction-2026-07-16/confirmation/aggregate.json`,
  `tasks/PORT-004-exact-backward-block-reduction.md`]
- **Dependencies**: []
- **Tags**: CUDA, backward, microprofile, performance, benchmark-only, negative-result
- **From staging**: direct PORT-004 result audit

## C50: State-matched checkpoints improve the bounded Janelle safe schedule; event color does not

- **Statement**: On the clean-commit FIT-023 one-image/one-seed RTX-4090 development factorial,
  checkpoint-only improves foreground/boundary PSNR by `+0.5020/+0.5194 dB`, CVaR99/p99 MSE by
  `11.00%/19.40%`, and relative interior/boundary undercoverage by `43.57%/10.10%` versus the
  matched global control at `+9.61%` total time. Four earlier state-matched field+Adam snapshots
  are actually committed. Post-topology event color solve alone is worse than control on every
  protected quality/coverage metric and `7.77%` slower end to end; combined is `29.26%` slower
  than checkpoint-only and does not Pareto-dominate it. Checkpoint-only is the bounded Janelle
  development recommendation; defaults remain unchanged.
- **Status**: supported single-image development mechanism; event-color promotion refuted on this
  image; general default unauthorized
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: Cold rescoring changes the stored ordering, the source/config
  equality audit fails, or a frozen multi-image/multi-seed comparison shows checkpoint-only does
  not retain its protected-metric advantage. A broader confirmation would strengthen or refute
  generality without rewriting this source-bound image result.
- **Proof**: [`ara/evidence/fit023-transactional-candidates-janelle-2026-07-23/run.md`,
  `runs/janelle_C0001_transactional_candidates_factorial_20260723/comparison.json`,
  `runs/janelle_C0001_transactional_candidates_factorial_20260723/audit.json`,
  `runs/janelle_C0001_transactional_candidates_factorial_20260723/report.md`]
- **Dependencies**: [C18, C22]
- **Tags**: safe-schedule, checkpoint, color-solve, mask-boundary, development, negative-result,
  claim-boundary
- **From staging**: direct FIT-023 result audit
