# Task index

Active work stays in `tasks/`; retired completed work lives in `tasks/done/`. Areas: CORE, INIT,
FIT, HIER, BENCH, ABL, FF, GEN, COMP, PORT, MERGE, DOCS. Work items are picked up via the
`structsplat-task-workflow` skill. `SESSION-BRIEF.md` is the generated session-start view of this
table; it never supersedes the Index.

The table below is the current outcome authority. Executed tasks whose protocol text is bound by
an artifact may intentionally retain a pre-execution `Status` section so its frozen hash remains
replay-valid; do not rewrite those task bytes merely to duplicate this table.

## Active Tasks

| ID | Title | Status | Depends on |
|----|-------|--------|-----------|
| ABL-001 | Init-strategy x budget sweep (the core experiment + fitness) | partial | INIT-003/004, BENCH-001 |
| ABL-002 | Full stage-combination search | partial | CORE, INIT, FIT, HIER, BENCH |
| GEN-001 | Generative 2D Gaussians via SDS distillation (no dataset) | todo | CORE-001, ADR-0006 |
| COMP-001 | Quantization + entropy/VQ codec (rate-distortion) | partial | FIT-001 |
| PORT-001 | CUDA tile rasterizer → IntrinsicEngine RHI pass | partial | CORE-001 |
| ABL-004 | Killer controls + full ABL-001 run + committed evidence | partial | BENCH-002, ABL-003, FIT-004 |
| COMP-003 | Compression-ratio ladder (scale ranges → planes → LSQ → VQ → entropy) | partial | COMP-002, BENCH-002 |
| CORE-007 | Segmentation-free responsibility boundary flux | design-only — BENCH-007 did not authorize the old structure/compression path; any reopening needs a new prior-art-controlled question | INIT-004, CORE-001, BENCH-007 |
| CORE-008 | Hybrid Gaussian + frequency-bearing primitive control | todo — conditional and off the Field V2 critical path; start only for a BENCH-022 residual with WIPES as the direct primitive control | BENCH-022/025, COMP-013/014, CORE-013, BENCH-002/007 |
| INIT-009 | Progressive WSE survivor ordering | implemented/confirmed — 32/32 uniform Euclidean prefix wins with identical terminal sets; opt-in for compatibility | INIT-003/005/006, BENCH-002 |
| PORT-002 | GPU-native tile index + fused loss/backward | partial — index/kernel work correctness-validated and its **preregistered profile passed** (RTX 3050, 2026-07-25, under the ADR-0024 parity amendment) — representative step ratio `0.6308` vs exact `cuda`, all 8 high-N grid cells ≤ 1.00, GPU index share `1.36%`; authorizes the fair-protocol end-to-end fit benchmark only, default stays `cuda`. Fused loss and CUDA graphs still open | PORT-001, FIT-003, ADR-0024 |
| PORT-003 | Avoid atomics in tiled backward | partial — gradient-validated and measured under PORT-002's passed profile — warp-reduced tiled backward is uniformly faster than exact `cuda` (e.g. 512²/N=8192/ov16/ar6 `5.960 → 2.110 ms`); no default flip authorized | PORT-001, ADR-0024 |
| PORT-004 | Exact-backward block reduction | implemented/screened — large N=2048 microprofile gains, but all-grid direction and independent CV guards fail; benchmark-only, not CLI/default | PORT-001/003, ADR-0011, BENCH-002 |
| GEN-003 | VSD / multi-particle distillation | todo | GEN-001 |
| ABL-005 | Fitter-knob influence pass at the fair regime | partial — CUDA-native fair shard started; color-solve and broader Kodak cells pending | ADR-0010, FIT-005/006/007, CORE-006 |
| BENCH-005 | Native external-reference pipelines and paired central metrics | partial — GI++/Image-GS/GaussianImage adapters plus bounded native GaussianImage/AIR evidence complete; full-resolution actual-RD and remaining 2026 methods open | BENCH-001/002/003, ABL-004 |
| FIT-013 | Geometry-consistent Sobel regularization | partial — quality candidate validated on COCO proxy and Kodak4; speed blocks promotion | FIT-005/006/007, ABL-004 |
| FIT-014 | Generation-density covariance filtering | implemented/screened — exact and weaker cohort filters lose the COCO4 proxy; default off | FIT-004, CORE-002, COMP-002, BENCH-002 |
| FIT-015 | Same-final-count best-PSNR checkpoint selection | implemented/confirmed — +0.4884 dB pooled across 72 Kodak trajectories; opt-in for sparse/moderate-density long fits | FIT-001/002, ABL-004, BENCH-005 |
| FIT-016 | Coarse-to-full loss-target curriculum | implemented/screened — rejected by 500-step guard (−0.1645 dB selected PSNR); default off | FIT-015, HIER-003/004, ABL-004, BENCH-002 |
| FIT-017 | Kernel-matched signed-residual densification | implemented/screened — wider scores improve immediate PSNR but lose post-20/post-100; default legacy score retained | FIT-004, FIT-009, BENCH-002, ABL-004 |
| FIT-018 | Responsibility-normalized error-density densification | implemented/screened — SAD alpha-0.7 transfer rejected (−0.0198 dB post-20, 4/8 wins); opt-in control only | FIT-004, FIT-009, FIT-017, BENCH-002, ADR-0010 |
| FIT-019 | Opacity-split gauge-equivalence audit | implemented/screened — exact commutation confirmed, recovery utility rejected; no production quotient/lineage state | FIT-007, FIT-009, FIT-018, BENCH-002 |
| FIT-020 | Ranked deduplication response spectroscopy | implemented/screened — signal strong, but bend prediction and screening rejected; close lineage, no production selector | FIT-007, FIT-009, FIT-018/019, BENCH-002 |
| BENCH-006 | Fixed-storage all-method convergence lane | completed as a 320-cell high-rate local-policy diagnostic; superseded for compression decisions by BENCH-007 | BENCH-001/002/003/004, COMP-002, ABL-004 |
| BENCH-007 | Actual-rate structure phase diagram | completed negative — Stage-1 gate failed; Stage 2 prohibited and not run | BENCH-001/002/003/004/006, COMP-001/002/004, INIT-003/009 |
| BENCH-008 | Common/native causal bridge | not authorized — BENCH-007 found no promotable renderer/objective interaction | BENCH-005/007, CORE-001/003/005, COMP-002 |
| BENCH-009 | Rate/DOF-priced residual tangent-space auction | completed negative/unavailable — exact Stage-1/recovery ledgers complete; causal calibration and projector validity failed; no method/expressiveness claim | CORE-001/006, FIT-005/017/020, BENCH-002, COMP-006 |
| BENCH-011 | Nested residual-extension spent-data diagnostic | completed negative — corrected v2 exactly reproduces BENCH-009 bases and all 96 rows; all four calibration strata fail; close without retuning | BENCH-009 |
| BENCH-012 | Spatial-connectivity policy value for finite reallocation | completed unavailable — topology core passed, but first preflight cell had only 2/4 required untruncated equal-work actions; no selector or recovery outcome scored | BENCH-009/011, FIT-004/017/020, COMP-006 |
| COMP-007 | Gauge-free log-Euclidean covariance codec | completed negative — v4 audit replayed all 12,096 streams; log-SPD failed 7/8 frozen gates and confirmation stayed sealed | COMP-006, BENCH-012 |
| COMP-008 | Mean-conditioned entropy-oracle killing test | completed inconclusive — both fixed tuples survived the necessary-condition lower-bound screen; this authorized COMP-009 only, not an actual compression claim | BENCH-016, COMP-006/007 |
| COMP-009 | Exact SSP2E actual-coder development test | completed negative — both tuples failed actual-rate and spatial-attribution gates; frozen decision `ABANDON_FIXED_SSP2E_V1`, confirmation sealed | COMP-008 |
| COMP-010 | SSP2E captured-replay relocation repair | completed provenance GO — captured-source replay confirms COMP-009's frozen negative decision; no new rate, quality, convergence, or performance evidence | COMP-009 |
| COMP-011 | Complete-stream RGB VQ/RVQ | terminal invalid/no-decision — candidate-byte replay passed, but one original decision-relevant LPIPS constituent is inside the frozen replay exclusion band; no SSP2V/SSP2L confirmation | COMP-008/009/010 |
| COMP-012 | Exact-byte RGB coordinate RDO with unchanged SSP2F | repaired draft/pre-data — objective/search/oracle/operator/paired-adapter cores implemented and hostile-tested; lifecycle orchestration, runtime/transcript caps, frozen gates, and preflight remain before any target access | COMP-011 provenance audit, COMP-009 SSP2F |
| BENCH-013 | Local-linear reproducing compositor | completed negative — affine reproduction passed, but signed local leverage/ringing killed 82/108 forward cells | COMP-007, CORE-006 |
| BENCH-014 | Explicit affine carrier | completed negative — strong synthetic quality/rank/cost, but complete-byte and terminal-convergence gates failed | BENCH-013, CORE-006 |
| BENCH-015 | Decoder-synchronized affine lift | completed negative — equal-byte smooth quality passed; no-harm, convergence, and cold-decode gates failed; Stage 1/local successor prohibited | BENCH-014, COMP-007 |
| BENCH-016 | Pinned native SAD frontier screen | completed negative development screen — valid 144-row v6 matrix; 0.5-bpp gates passed, 2.0-bpp quality/LPIPS guards failed; decision `abandon SAD reuse` | BENCH-005/007/015, FIT-018 |
| COMP-005 | Decoder-synchronized structural geometry | not authorized — tensor structure failed the actual-rate gate and layout bytes were not established as the binding loss | BENCH-007, COMP-001/002/003/004, INIT-001/003/009 |
| COMP-006 | Marginal cold-stream rate--distortion attribution | completed negative — exact replay matched; birth lost `-1.0714 dB` to the strongest actual-byte control and confirmation remains sealed | COMP-001/002/003/004, BENCH-002/007, FIT-004/017 |
| CORE-010 | Mask-contained fitting for alpha-masked inputs | implemented (opt-in, default off) — SDF hard containment (mean projection + dynamic caps, ADR-0017) + out-of-mask coverage penalty + masked loss; exact zero outside with support_fade; directional-cap follow-up landed as CORE-011; five-arm cost benchmark + pyramid support deferred | CORE-003/005, INIT-002/003, FIT-012, ADR-0012/0017 |
| CORE-011 | Boundary coverage for mask-contained fitting | implemented (opt-in, default off) — certified anisotropic tangent caps (station-ball SDF certificate, ADR-0019) + boundary-band under-coverage hinge + boundary tangent densification + CLI band PSNR; exact-zero-outside guarantee preserved; committed multi-arm benchmark deferred (shared with CORE-010) | CORE-010, ADR-0017/0019, CORE-003/005, FIT-012 |
| PORT-005 | Batch encode throughput (multi-process across images) | implemented (CPU-validated) — `batch-fit` CLI with worker pool, device round-robin, resumable metrics.jsonl, failure isolation; GPU scaling row pending | FIT-001, CORE-001, ADR-0011, PORT-001 |
| FIT-021 | Pooled row lifecycle, byte-budgeted capacity, error-triage events | implemented (opt-in, default off, ADR-0020) — fixed-capacity pool with off-image parking + free list, capacity from `target_file_bytes` via SSPL1 raw layout + in-container alpha stream, one triage event (responsibility-gated park → envelope merge → split → spawn) with in-place optimizer rows; CPU end-to-end + budget/determinism tests; benchmark slice and blob-dispatch/color-fixable phase 2 open | FIT-002/004/007/017/018, CORE-003/009/010/011, COMP-001, ADR-0007/0010/0017/0019/0020 |
| FIT-022 | Coverage-matching regularizer (feature-targeted fit-time Gaussian blue noise) | implemented (opt-in, default off) — mass-neutral `mean_inside (S−c)²` on the compositor's raw weight sum with detached opacities, tensor/tensor_boundary/error_blend targets, cosine decay; gradient/transport/compose tests green; the audit's preregistered fixed-N killing experiment (incl. additive-differential arm) remains the required screen before any promotion | 2026-07-23 ideation audit, CORE-010/011, FIT-012/021, ADR-0003/0010/0020 |
| FIT-023 | Transactional safe-schedule checkpoint/color candidates | completed (single-image development) — 2x2 Janelle factorial selects state-matched Pareto checkpoints every 50 steps; event color and combined are not promoted; defaults unchanged, multi-image confirmation open | FIT-005/010/015/021, CORE-010/011, BENCH-002 |
| FIT-024 | Transactional fixed-capacity storage | implemented (opt-in, A/A-calibrated single-image CUDA check) — safe-schedule topology/gates run over a preallocated active prefix with capacity-shaped Adam/checkpoints, active-shape update kernels, and terminal compaction; no quality regression beyond dynamic A/A resolution was detected, peak allocated GPU memory was ~11.5 MiB lower, runtime was neutral, default remains dynamic pending multi-image evidence | FIT-021/023, CORE-010/011, ADR-0020/0021 |
| FIT-025 | Reserved error-adaptive detail tail | implemented and development-screened (opt-in, ADR-0022) — physical capacity is separate from the ordinary active ceiling; bounded covered-interior high-frequency tail births/splits retain the full safe gate and gain-per-row stop. On the matched 12,024-row fixed-pool Janelle screen, generic +512 strictly beats the equal-count specialized tail on every nontrivial protected metric and is fastest observed; specialized tail/default promotion rejected | FIT-023/024, CORE-010/011, ADR-0021/0022 |
| FIT-026 | Geometric-growth storage policy | implemented (opt-in, default off, ADR-0023) — `storage_policy="geometric"` starts the transactional pool at `initial_capacity` and grows physical capacity by `growth_factor` toward `capacity` on demand (append parked rows + `adapt_optimizer_state`), O(log N) migrations; the live prefix is preserved bit-for-bit so a geometric fit is `atol=0`-identical to `fixed_capacity` in field and metrics; ported from realtime-gs `GeometricParameterArena`; timing vs fixed/dynamic left to a FIT-024-style benchmark | FIT-021/023/025, CORE-010/011, ADR-0020/0021/0022/0023 |
| FIT-027 | Cached target-side SSIM statistics | implemented — `metrics.SSIMTargetStats` plus an opt-in `target_stats=` argument; `fit()` owns one cache per constant curriculum target (matted once when a mask is present) and leaves the blended lerp state uncached. Identity + autograd version-counter guard, so a swapped or in-place-mutated target invalidates rather than poisons the loss; `metrics.ssim()` stays stateless when no cache is passed. Fit trajectories are bit-identical on both arms (test_fit_dynamics). GPU speedup unmeasured — CPU microbench shows `1.27--1.75x` on the SSIM term | PORT-002, FIT-003, FIT-016 |
| CORE-012 | One maintained entrypoint for the current best pipeline | implemented (ADR-0025/0028) — `structsplat.pipeline.run_pipeline` is the canonical recipe and `scripts/convert.py` the sole conversion CLI; masked/full-frame arms share one safe-commit schedule and the full-frame `FitConfig` validation path is repaired; the `0.75` mask margin reproduces the executed Janelle recipe (C56), without a comparative margin claim; full-frame arm unscreened pending BENCH-017 | FIT-023/024/025, CORE-010/011, INIT-009, ADR-0013/0017/0019/0025/0028 |
| BENCH-017 | Screen the full-frame pipeline arm against the plain-fit path | todo — preregistration required; decides whether unmasked `convert` keeps the schedule or dispatches to plain fit | CORE-012, ADR-0025, BENCH-002, ABL-006, FIT-015 |
| FIT-028 | Measure the interior coverage budget and decide its default | todo — mechanism shipped (ADR-0026) with default `0.0`, so behaviour is unchanged; the screen that would authorize a nonzero default has not run. Motivated by O87: 82 of 110 rejected blocks died on `interior_holes_regressed` alone | ADR-0026, ADR-0025, BENCH-002, BENCH-017, FIT-023 |
| FIT-029 | Decide whether `safe_polish` earns its wall-clock | todo — **blocked on FIT-028 by design**; 0 of 3,276 attempted steps committed across 7/7 unmasked images (O88), but all 7 rejections were the shared interior-hole veto, so veto-driven and tolerance-driven causes are not yet separated | FIT-028, ADR-0026, FIT-023, BENCH-017 |
| FIT-030 | Byte-priced topology and precision allocation | todo — use estimated marginal gain only for proposals; select and stop on the chosen complete codec bytes and cold-decoded quality | BENCH-021/025, COMP-013/014, FIT-045/046, BENCH-002 |
| FIT-031 | Error-only fine-detail tail | implemented/screened — one exposed Janelle run accepted 4,608/7,089 requested rows and improved its own pre-tail protected metrics, but the existing default is not count/rate/trajectory matched; default remains off (C58) | FIT-023/025, CORE-012, ADR-0022/0025/0028/0029 |
| FIT-032 | Gauge-lifted residual dipoles | completed negative — 0/3 frozen budgets pass; ordinary births remain stronger | FIT-007/017/019/020/025/031, BENCH-002, CORE-012, ADR-0029 |
| FIT-033 | Residual-birth partial color solve | completed exposed-image positive — exact frozen-base solve reaches 6.47% deep high-pass reduction at 128 rows; independent confirmation open | FIT-005/017/025/031/032, CORE-012, BENCH-002 |
| FIT-034 | Spectral partial color solve | completed negative — 6.55%/7.51% high-pass/Laplacian at 128 rows misses the frozen gate | FIT-005/017/025/031/032/033, CORE-012, BENCH-002 |
| FIT-035 | Sparse affine detail births | completed negative — best protected 128-row arm reaches 8.90%/9.72% but misses the gate and costs extra coefficients | CORE-006/012, FIT-005/017/025/031/032/033/034, BENCH-002/009/011 |
| FIT-036 | High-pass residual-ridge births | completed negative — residual anisotropy stays near the isotropic result at 6.57%/7.51% | FIT-017/025/031/032/033/034/035, CORE-010/011/012, BENCH-002 |
| FIT-037 | Minimum-row fine-detail target | completed negative — static ranking reaches only 15.01%/12.04% at 2,048 rows | FIT-031/033/034/035/036, CORE-012, BENCH-002 |
| FIT-038 | Orthogonal fine-detail pursuit | completed negative control — 5x5 cross-wave exclusion reaches 20.22%/16.21% at 2,048 rows | FIT-031/033/037, CORE-012, BENCH-002 |
| FIT-039 | Detail-pursuit exclusion radius | completed exposed-image positive — exact-site-only exclusion first reaches 25%/20% at 768 rows; no default promotion | FIT-038/037/033, CORE-012, BENCH-002 |
| FIT-040 | Opt-in orthogonal fine-detail pursuit tail | implemented/screened — `--fine-detail-pursuit` reproduces the 768-row winner with protected-safe production telemetry; default remains off (C59/C60) | FIT-005/031/033/038/039, CORE-010/011/012, BENCH-002, ADR-0029/0030 |
| FIT-041 | Equal-base error-tail control | completed — FIT-031 wins global PSNR, pursuit wins deep detail and LPIPS with 3.616x fewer rows on one exposed full frame (C60) | FIT-031/039/040, CORE-012, ADR-0029/0030 |
| FIT-042 | Independent fine-detail pursuit confirmation | todo — preregistered held-out killing screen plus sealed confirmation; tests FIT-040 against equal-row generic/static controls and FIT-031 as an objective control, with actual-byte/work accounting; no default change authorized | FIT-031/033/037/038/039/040/041, CORE-012, BENCH-001/002, COMP-001, ADR-0029/0030 |
| FIT-043 | Sequential error-then-pursuit tail | completed negative — 51/51 cumulative target hits and rules 2–4 pass, but exact persisted error-prefix equality is 43/44 because one `scale_max` certificate refreshes at stage entry; frozen rule 1 rejects without retuning, no production/default or FIT-042 claim | FIT-031/040/041/042, CORE-012, BENCH-002, ADR-0029/0030 |
| FIT-044 | Stage-wise parameter-group activation schedules | todo — component screen for the selected Field V2 semantics | FIT-016, BENCH-020, CORE-013, BENCH-002 |
| FIT-045 | Direct-control regional allocation and merge screen | todo — fixed-N/global/uniform/LocoADC-style controls; no novelty claim | FIT-017/018, BENCH-020, CORE-013, BENCH-002 |
| FF-002 | Field V2 Predict–Optimize–Distil | todo — optional amortized lane after field, recipe, and codec selection | FF-001, CORE-013, BENCH-020/021/025, COMP-013/014, BENCH-002 |
| FF-003 | Complete-byte elastic Field V2 predictor | todo — one checkpoint over a complete-byte ladder | FF-002, BENCH-025, COMP-013/014 |
| BENCH-018 | Commit-gate granularity (`block_steps`) | todo — no code change needed, `--block-steps` already exposed; overall acceptance ~9% with `detail_growth` at 6.2% over ~56% of wall-clock (O88). Interacts with FIT-028 and FIT-027, so sequence after both | BENCH-002, BENCH-017, FIT-023, ADR-0025 |
| BENCH-019 | Stage-1 downstream-objective validity | partial — both passive protocol/report and realtime-gs receipt-export substrates are implemented; exporter review, adapters/predictor collection, complete matched fields, distinct protocol review, and formal outcome remain open | BENCH-001/002, CORE-012 |
| CORE-013 | Observation Field V2 semantic contract | in-review — implementation and full gate clean at tree `1ec3245`; distinct semantic/architecture review required before acceptance | BENCH-019, CORE-001/002, COMP-002, ADR-0006 |
| BENCH-020 | Field semantics and alpha-policy factorial | partial — sealed fixed-geometry/factorial/confirmation/report substrate and external receipt exporter implemented; exporter acceptance, matched disjoint fields, exact executor profiles, distinct reviews, formal outcome, ADR, and ARA disposition remain open | BENCH-019, CORE-013, BENCH-002, ADR-0003/0006 |
| INIT-010 | Field V2 initializer transfer screen | todo — re-test deterministic geometry priors under selected semantics and matched early-fit work | BENCH-020, CORE-013, INIT-003/005/006/009, BENCH-002/004 |
| FIT-046 | Additive appearance variable projection | todo — matrix-free conditional coefficient solve and matched convergence screen | BENCH-020, CORE-013, FIT-005/010, BENCH-002 |
| FIT-047 | Unbiased tile-sampled fitting | todo — probability-recorded inverse-propensity estimator with full-objective checks | BENCH-020, CORE-013, BENCH-002, ADR-0024 |
| FIT-048 | Additive scale/topology stage-order screen | todo — full-N/single-scale versus progressive/coarse-to-fine under exact work accounting | BENCH-020, CORE-013, INIT-010, FIT-049, HIER-003/004, BENCH-002/004 |
| FIT-049 | Field V2 objective and loss screen | todo — isolate RGB, structural, and downstream terms only after semantic selection | BENCH-019/020, CORE-013, FIT-012/016, BENCH-002 |
| BENCH-021 | Additive convergence portfolio | todo — successive-halving composition gate for initializer/loss/stage and FIT-044/045/046/047 | BENCH-020, INIT-010, FIT-044/045/046/047/048/049, BENCH-002/004 |
| COMP-013 | Observation Field V2 codec | todo — complete bytes, target-rate control, cold decode/query, strict versioning | CORE-013, BENCH-020, COMP-002/004/008/009, BENCH-002 |
| PORT-006 | Additive end-to-end acceleration | todo — reference parity plus representative conversion speed, not kernel timing alone | BENCH-020/021, FIT-046/047, ADR-0011/0024 |
| CORE-014 | Additive production pipeline (default-off) | todo — integrate only the frozen winners under the sole conversion entry point | CORE-013, BENCH-021/025, COMP-013/014, FIT-030, PORT-006, ADR-0006 |
| BENCH-022 | Additive production confirmation | todo — sealed complete-pipeline go/no-go against native-additive and normalized controls | CORE-014, BENCH-019/020/021/025, COMP-013/014, PORT-006, BENCH-002 |
| CORE-015 | Promote Observation Field V2 conditionally | todo — change defaults only on BENCH-022's exact positive profile digest; otherwise abandon | BENCH-022, CORE-014 |
| BENCH-023 | Amortized encoder confirmation | todo — held-out Field V2 quality/rate/latency and training break-even | FF-002/003, BENCH-022/025, COMP-013/014, BENCH-002 |
| BENCH-024 | Temporal field-reuse killing test | todo — same-camera warm-start/shared-geometry/delta opportunity only | CORE-014, COMP-013/014, BENCH-022/025, BENCH-002 |
| BENCH-025 | Structured-codec necessity gate | todo — test whether seed-generated local structure beats the complete direct codec at usable cold-query cost | BENCH-020, COMP-008/009/013, CORE-013, BENCH-002 |
| COMP-014 | Seed-structured Field V2 codec (conditional) | todo — implement one SGI-controlled grammar only after a positive BENCH-025 verdict; otherwise close without code | BENCH-025, COMP-013, CORE-013, BENCH-020, COMP-008/009 |
| DOCS-004 | Staged lint/format ratchet (widen `select`, adopt `ruff format`) | todo — Stage 1/2 due before the next results-bearing task closes; see the task's Expiry section | DOCS-003 |
| DOCS-006 | Repository-native experiment workflow skill | in-progress | DOCS-005, BENCH-002/003 |

## Proposed Additive Observation Field V2 execution order (2026-08-03)

The binding proposal and rationale are in `docs/additive_field_v2.md`. This is an evidence-gated
plan, not a default decision or a measured claim:

1. **Contract gate:** BENCH-019 → CORE-013 → BENCH-020. Stop if no semantic candidate is valid and
   downstream-favorable.
2. **Convergence gate:** run INIT-010 and FIT-044/045/046/047/048/049 from the selected contract,
   then compose only preregistered interactions in BENCH-021.
3. **Rate and implementation gate:** build direct COMP-013 and PORT-006 around the selected
   semantics; BENCH-025 decides whether direct coding is enough, and COMP-014 implements one
   seed-structured grammar only on a positive verdict. FIT-030 prices actions through the selected
   complete codec.
4. **Integration gate:** CORE-014 assembles one default-off profile; BENCH-022 gives the sealed
   end-to-end go/no-go; CORE-015 may change the default only after a positive verdict.
5. **Optional amortization/research:** FF-002 → FF-003 → BENCH-023, plus BENCH-024 and CORE-008,
   remain outside the production critical path.

BENCH-017, FIT-028/029, and BENCH-018 continue to answer normalized transactional-pipeline
questions, but no longer block the additive proposal. FIT-042 remains its independent normalized
fine-detail confirmation.

## Retired Done Tasks

| ID | Title | Path |
|----|-------|------|
| CORE-001 | Differentiable reference rasterizer (normalized weighted sum) | `done/CORE-001-reference-rasterizer.md` |
| CORE-002 | RS Gaussian parameterization + conics | `done/CORE-002-rs-gaussian-params.md` |
| CORE-003 | Edge-aware render support window (off-image support + tile waste) | `done/CORE-003-render-support-clamp.md` |
| CORE-004 | Renderer + GaussianField correctness fixes (CUDA N=0, int-cast UB, aliasing, dilation) | `done/CORE-004-renderer-field-correctness.md` |
| CORE-005 | Reference renderer memory bound + C0-continuous support cutoff | `done/CORE-005-renderer-memory-continuity.md` |
| CORE-006 | Linear color basis per Gaussian | `done/CORE-006-affine-color-basis.md` |
| CORE-009 | DC / background layer under the detail Gaussians | `done/CORE-009-dc-background-layer.md` |
| INIT-001 | Structure tensor: energy, orientation, flat/edge/corner | `done/INIT-001-structure-tensor.md` |
| INIT-002 | Density field (image + residual) | `done/INIT-002-density-field.md` |
| INIT-003 | Anisotropic blue-noise sampling (WSE + metric) | `done/INIT-003-anisotropic-pds.md` |
| INIT-004 | Flanking vs on-edge placement + threshold study | `done/INIT-004-flanking-vs-onedge.md` |
| INIT-005 | Init-math robustness, flanking unification, WSE test coverage | `done/INIT-005-init-robustness.md` |
| INIT-006 | Init-time performance (quadtree, spacing, run-lengths, pair discovery) | `done/INIT-006-init-performance.md` |
| FIT-001 | Adam fitter (L1+SSIM), PSNR history, iters-to-target | `done/FIT-001-optimizer.md` |
| FIT-002 | Fitter correctness (split colors, opacity pruning, history pairing) | `done/FIT-002-fitter-correctness.md` |
| FIT-003 | Fit-loop speed (device-side targets, SSIM hygiene, fused SSIM) | `done/FIT-003-fit-loop-speed.md` |
| FIT-004 | Densification & convergence upgrades (fp-growth, relocation, NMS) | `done/FIT-004-densification-upgrades.md` |
| FIT-005 | Exact / alternating color solve | `done/FIT-005-exact-color-solve.md` |
| FIT-006 | Frequency-violation densification | `done/FIT-006-frequency-violation-densification.md` |
| FIT-007 | Moment-preserving split / clone | `done/FIT-007-moment-preserving-split.md` |
| FIT-008 | Self-adaptive Gaussian count | `done/FIT-008-self-adaptive-gaussian-count.md` |
| FIT-009 | Factor the refine axis into orthogonal sub-axes | `done/FIT-009-factor-refine-axis.md` |
| FIT-010 | Cheap color-solve schedules | `done/FIT-010-color-solve-schedules.md` |
| FIT-011 | Split-recovery micro-levers (moment seeding, warmup, scheduled fade) | `done/FIT-011-split-recovery-microlevers.md` |
| FIT-012 | Edge-weighted pixel loss (structure-tensor loss weighting) | `done/FIT-012-edge-weighted-loss.md` |
| FF-001 | Feed-forward init predictor (warm-start) | `done/FF-001-feedforward-predictor.md` |
| HIER-001 | Progressive pyramid (residual-driven densification) | `done/HIER-001-progressive-pyramid.md` |
| HIER-002 | Pyramid bookkeeping (iteration accounting, budgets, schedules) | `done/HIER-002-pyramid-bookkeeping.md` |
| HIER-003 | Pyramid equal-iteration diagnosis (fix or retire HIER-001) | `done/HIER-003-pyramid-iteration-accounting.md` |
| HIER-004 | Pyramid convergence repair and promotion decision | `done/HIER-004-pyramid-convergence-repair.md` |
| BENCH-001 | Metric protocol (PSNR/MS-SSIM/LPIPS + iters-to-target) | `done/BENCH-001-metrics.md` |
| BENCH-002 | Benchmark harness experimental-validity fixes (equal budgets, resumable sweeps, seed-aware comparisons) | `done/BENCH-002-harness-validity.md` |
| BENCH-003 | Benchmark script consolidation + documentation | `done/BENCH-003-benchmark-consolidation.md` |
| BENCH-004 | Sweep-cost controls (plateau exit, multi-target tables, proxy regime) | `done/BENCH-004-sweep-cost-controls.md` |
| ABL-003 | Bisect the undiagnosed −0.794 dB flagship regression | `done/ABL-003-regression-bisect.md` |
| ABL-006 | Successive-halving execution of the remaining confirmation | `done/ABL-006-successive-halving-confirmation.md` |
| INIT-007 | Retire the flanking default (measured answer + ADR) | `done/INIT-007-retire-flanking-default.md` |
| INIT-008 | Feature-relative scale caps (fix the cap-scaling failure) | `done/INIT-008-feature-relative-scale-caps.md` |
| MERGE-001 | Integrate Claude core optimizations and Codex stage search into main | `done/MERGE-001-claude-codex-main.md` |
| COMP-002 | Codec / metrics / CLI correctness and protocol fixes | `done/COMP-002-codec-correctness.md` |
| COMP-004 | QAT + entropy-aware fitting | `done/COMP-004-entropy-aware-fitting.md` |
| DOCS-001 | Docs-sync backfill (stale status, missing ADRs, ara scaffold) | `done/DOCS-001-docs-sync-backfill.md` |
| DOCS-002 | Publication visual diagnostics | `done/DOCS-002-publication-visual-diagnostics.md` |
| DOCS-003 | Verification spine (docs_sync + verify.sh + markers/seeding + CI) | `done/DOCS-003-verification-spine.md` |
| DOCS-005 | Agentic workflow maturity | `done/DOCS-005-agentic-workflow-maturity.md` |

Retired tasks remain valid dependency IDs. They describe completed reference/correctness work; the
performance and scale follow-ups stay active under PORT/FIT/INIT/BENCH/ABL tasks.

## Current priority after the 2026-07-15 recovery and marginal-RD assays

FIT-018--020 completed the recovery branch requested after BENCH-007, and COMP-006 completed its
next exact-rate screen. Responsibility-density transfer, opacity-gauge quotient allocation, the
frozen response bend, and one-extra-standard-row at a complete-stream byte cap all failed their
promotion gates. Their exact grouping, response logging, and cold-stream selection machinery
remain benchmark oracles, not production features. The queue is therefore:

1. Preserve the FIT-018--020 and COMP-006 negative results; do not retune their fixtures, splits,
   alphas, bends, horizons, caps, actions, bit box, or exposed held-out targets.
2. Preserve BENCH-009's completed v3 ledgers and failed causal decision. Do not reinterpret its
   negative incremental projector rows, late carrier recovery, objective mismatch, or provisional
   packet bytes as expressiveness, convergence, performance, or compression evidence.
3. Preserve BENCH-011 v1 as an audited invalid run and corrected v2 as the valid negative. V2
   reproduces BENCH-009's exact base seeds/ranks and all `96` rows, but all four calibration strata
   fail. Close the local-linear extension formulation without retuning or disjoint-data spending.
4. Preserve BENCH-012 as an unavailable preflight, not a topology result. Do not relax its
   equal-work/untruncated candidate filter, move the exposed targets, shrink its inherited action,
   or reinterpret the 2/4 feasibility failure as quality, convergence, or allocation evidence.
5. Preserve PORT-004's opt-in exact-backward block-reduction prototype as benchmark-only. It
   repeats large `N=2048` representative gains, but fails the governing all-grid direction rule
   and an independent run's `5%` stability guard; do not open end-to-end or default promotion.
6. Preserve COMP-007 v2/v3 as pre-scoring unavailable artifacts and v4 as the valid negative. Do
   not retune the log-SPD chart or expose the odd Kodak IDs; its `0.3426%` zstd median movement is
   below the frozen effect and seven of eight gates fail.
7. Preserve BENCH-013--015 as the closed first-order-reproduction lineage. The local-linear
   compositor, transmitted affine carrier, and decoder-synchronized lift each failed their frozen
   gate; do not run Stage 1 or retune a local successor on these artifacts.
8. Explore existing-grammar attribute/rate co-design—per-group QAT, real context/range coding, and
   `R + lambda D`—as a separate task that retains COMP-006's exact-byte gate.
9. Continue BENCH-005 native-authentic coverage only as external-validity infrastructure; it cannot
   retroactively promote a failed local mechanism.

## Dated priority after the 2026-07-14 actual-rate decision

BENCH-007 is complete. Tensor-WSE showed a bounded 0.5-bpp gain over the strongest local gradient
control, but the effect vanished at 1.0 bpp, reached only `-4.5417%` BD-rate versus the required
`-10%`, cost `1.4752x`, and exceeded the texture guard. The preregistered gate failed, so Stage 2,
BENCH-008, and COMP-005 are not authorized. The queue is therefore:

1. Preserve the BENCH-007 negative result and do not tune the eight Stage-1 images or consume
   untouched DIV2K validation as a rescue set.
2. Close the current tensor-WSE compression-paper claim. F5--F9 and the actual-rate harness are
   reusable infrastructure, not held-out method evidence.
3. If method research continues, begin a fresh `structsplat-research-ideation` pass around a
   materially different question, null, hard compute/texture guards, and a disjoint development
   screen. The low-rate edge/bleed signal may motivate that search but cannot preselect a rescue
   formulation on the failed pilot.
4. Continue BENCH-005's native-authentic coverage only as an independent external-validity or
   benchmark lane; it cannot retroactively promote tensor-WSE.
5. Keep CORE-007/008 design-only until a new question and their direct Contour-Aware 2DGS/WIPES
   controls justify implementation.

The dated priority sections below are retained as execution history, not as the current queue.

## Suggested order (from the 2026-07-03 repo review)

CORE-004/FIT-002/HIER-002/COMP-002/INIT-005/BENCH-002/ABL-003 fix confirmed bugs and
science-gating ambiguities. FIT-003 removed fit-loop metric overhead and added the optional fused
SSIM backend, and FIT-004 added the densification/relocation controls needed for the experiment.
Next run ABL-004 (the actual experiment, with evidence committed). After that, the 2026-07 SOTA
review suggests the most pragmatic improvement order:

1. FIT-005 exact/alternating color solve — completed 2026-07-06; keep default off, stage-search
   axis available as `color_solve=every10`.
2. FIT-006 frequency-violation densification — completed 2026-07-06; keep default off,
   stage-search refine axis available as `freq_violation`.
3. FIT-007 moment-preserving split — completed 2026-07-06; keep default off,
   stage-search refine axis available as `moment_preserving`.
4. CORE-006 affine color basis — completed 2026-07-06; keep default `constant`, stage-search
   axis available as `color_basis=affine`.
5. FIT-008 self-adaptive Gaussian count — completed 2026-07-06; keep default fixed-N, but
   `--adaptive-count` now exposes target/max-N/stall-controlled growth and selected-N metadata.
6. FF-001 feed-forward teacher-student warm start — completed 2026-07-07. Tensor-prior inputs make
   the tiny predictor useful versus random scratch on a held-out slice, but it still loses to the
   hand `quadtree_wse` prior; keep `strategy=feedforward` experimental.
7. COMP-004 compression-aware fitting — completed 2026-07-07. Fit-time STE QAT and lambda sweeps
   are available, but post-fit QAT remains the stronger default on the local RD slice.
8. PORT-002/PORT-003 if tiled CUDA remains strategically important after quality work.
9. GEN-003 after GEN-001 has a debuggable SDS baseline.

## Suggested order (from the 2026-07-07 benchmark review)

The 2026-07-04..07 evidence answered the flanking question negatively (INIT-004; retirement is
INIT-007) and shifted the frontier from init strategies to fitter knobs and sweep economics:

1. BENCH-004 sweep-cost controls — completed 2026-07-07. Use the 512/750 proxy screen for cheap
   screening, keep early exit opt-in, and promote only from full fair-regime evidence.
2. ABL-005 fitter-knob influence pass — the +0.26 dB `charbonnier`/`variance`/`opacity` bundle
   and the FIT-005/006/007/CORE-006 candidates, isolated at the fair regime. Highest expected
   dB-per-GPU-hour in the queue.
3. ABL-006 successive-halving confirmation — completed 2026-07-07; fed INIT-007's default flip
   (ADR-0013), also completed 2026-07-07.
4. FIT-009 refine-axis factoring — completed 2026-07-07. The new
   `residual_tensor x moment_preserving` combination is now expressible but did not win the
   difficult-four fair slice; keep it as searchable, not default.
5. FIT-010 color-solve schedules — completed 2026-07-07. `on_split` helped split recovery but
   failed the final-PSNR promotion rule; keep `every<N>` as the quality arm and `on_split`
   searchable.
6. FIT-011 split-recovery micro-levers — completed 2026-07-07. State seeding and row tempering did
   not improve split recovery, and scheduled fade missed the AUC promotion rule; keep all three
   searchable, default off.
7. INIT-008 feature-relative caps — completed 2026-07-07. `feature_rel` repairs most of the old
   resolution-scaled cap failure but fails the no-loss promotion rule (48 paired cells, mean
   dPSNR -0.3733 vs uncapped); keep it searchable and default off.
8. FIT-012 edge-weighted loss — completed 2026-07-07. `loss_weight=tensor` is PSNR-neutral in the
   difficult-four aggregate (+0.0061 dB over 16 pairs) and helps `aniso_onedge`, but it hurts
   `quadtree_wse` and loses AUC on average; keep it searchable and default off.
9. HIER-003 pyramid diagnosis — completed 2026-07-07. The current two-level pyramid is a
   final-quality positive (+1.0000 dB mean PSNR over the difficult-four 2k/5k slice), but it loses
   AUC in every pair (-1.3540 mean), so HIER-004 owns convergence repair.
10. HIER-004 pyramid convergence repair — completed 2026-07-07. Explicit per-level schedules are
   available; 150/1350 repairs the AUC loss while preserving final PSNR on the difficult-four
   slice (+0.0601 dB vs the 750/750 pyramid control and +0.0011 AUC vs single). Keep `single` as
   shipped default pending larger confirmation; use 150/1350 as the pyramid quality candidate.
11. CORE-009 background layer — completed 2026-07-07. A counted frozen-geometry background
    Gaussian layer is a strong low-budget quality candidate (`frac0.05_grid8`: +1.0152 dB mean
    PSNR over 24 pairs), but it loses AUC at 5000 rows; keep it searchable and default off.
12. FF-001 — completed 2026-07-07 with multi-image teacher training and tensor-prior input
   ablation.
13. COMP-004 — completed 2026-07-07. Fit-time STE QAT helps low/mid-bit direct-encode quality,
   but the existing post-fit QAT control is still marginally better overall; keep the new knobs
   searchable, not default.
