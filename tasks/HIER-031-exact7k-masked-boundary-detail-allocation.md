# HIER-031 — Exact-7k masked boundary and thin-detail allocation

## Context

HIER-030 showed that exact hard containment succeeds at 7,000 rows, but the masked C0001
reconstruction is dominated by a narrow boundary-closure error band: 94--96% of foreground SSE
lies within four pixels of the mask edge and every zero-coverage pixel lies within 3.17 pixels of
that edge. The maintained masked pipeline already has tangent boundary initialization and
count-neutral boundary recycling, while FIT-040 has a deep high-frequency pursuit mechanism.
Neither has been tested as an exact-7,000-row allocation policy on this raster. Scaling the count
to the approximately 57,600-row density-equivalent regime is explicitly deferred.

## Goal

At exactly 7,000 persisted Gaussians, determine whether holes and thin structures can be repaired
by reallocating rows among ordinary interior, boundary/thin-mask topology, and high-frequency
appearance, while retaining exact zero support and reconstruction outside the mask.

## Frozen development protocol

- Exposed development input: HIER-030's hash-bound C0001 image/mask, deterministically resized to
  1200x1038, seed 0, CUDA, black-matted target, hard anisotropic containment, margin 0.75 px, and
  C0 compact-support fade. This is diagnostic evidence, not held-out confirmation.
- Every scored endpoint persists exactly 7,000 rows and only the four normal arrays. Every centre
  must be in the raw mask; maximum unit coverage and reconstruction outside must be <=1e-7.
- First perform a target-independent feasibility audit using the resized binary mask and current
  minimum scale. Report SDF bands, one-/two-pixel-thin topology, and the subset geometrically
  unreachable under the certified support radius. This audit may kill impossible arms but may not
  retune thresholds from reconstruction outcomes.
- Arms, in order:
  1. `pipeline_7k`: current masked safe schedule scaled to capacity 7,000.
  2. `boundary_recycle_7k`: the same schedule with its existing count-neutral boundary recycling
     enabled at capacity.
  3. `topology_reserve_7k`: reserve/recycle rows toward SDF-ridge (medial-axis) and thin-mask sites,
     tangent-aligned where defined; ordinary rows are reduced by the identical count.
  4. `detail_reserve_7k`: reserve 768 rows from the ordinary base and restore them with the frozen
     FIT-040 orthogonal high-pass pursuit, never exceeding 7,000.
  5. `combined_7k`: only if the first four establish complementary boundary-hole and deep-detail
     gains, combine their frozen operators within the same 7,000-row budget.
- A boundary-micro-scale causal arm may be added only if the feasibility audit proves that the
  0.35 px lower scale bound makes nonempty foreground pixels mathematically unreachable. It must
  be boundary/thin-site only, keep exact containment, keep N=7,000, and be labeled a representational
  repair rather than an allocation-only arm.
- Stage-2 donor-funding amendment (frozen after the first micro arm closed all holes but failed the
  interior guard): fund the identical certified micro-site operator by merging disjoint mutual-
  nearest ordinary pairs instead of deleting low column-energy rows. Select pairs with both centres
  at SDF>2 px by a fixed distance/color/log-scale/axial-angle score; replace each pair by its 1.05x
  covariance envelope, recertify ordinary rows anisotropically, spend the freed partner on the
  current raw-hole site, and re-solve all colors. Repeat for at most four waves only while raw holes
  remain. This is an exposed mechanistic rescue, not part of the original arm-ranking claim.
  The first implementation left previously inserted 0.08 px rows inside the ordinary 0.35 px
  recertification set on wave two and reopened its repaired sites. Preserve that failed execution;
  the intended replay exempts existing certified micro rows while recertifying only ordinary rows.
- Stage-3 thin-stroke amendment (frozen after corrected merge funding closed every hole but spent
  873 rows and missed the interior guard): place one candidate at each current raw-hole pixel with
  0.08 px across-scale, 2.0 px proposed along-scale, and the SDF-normal tangent; recertify using the
  existing station-ball anisotropic certificate at min-scale 0.08; greedily select candidates by
  number of still-uncovered hole pixels (stable site-index tie break) until every current hole is
  covered. Fund only that selected set with the same merge operator, exempt all existing certified
  micro rows, and repeat up to four waves while holes remain. No outcome-dependent scale, tangent,
  certificate ladder, or cover tie-break retuning.
- Stage-4 orientation-only killing control (frozen after the SDF-normal Stage-3 cover required 854
  rows for 869 holes because almost every long axis was capped): replace only the candidate angle
  by the principal axis of current hole/skeleton pixels in a fixed 7x7 neighbourhood, falling back
  to the Stage-3 SDF tangent when fewer than two neighbours exist. Keep the 0.08/2.0 scales,
  station-ball certificate, greedy cover, funding, and stopping rule identical.
- Stage-5 geometry-recovery amendment (frozen after both stroke covers remained near one row per
  hole): start from the corrected merge-funded zero-hole endpoint, freeze every <=0.081 px micro
  row in geometry and color, exempt it from the ordinary 0.35 px lower clamp/containment pass, and
  optimize the other 6,127 rows for the frozen HIER-030 500-step masked additive objective with no
  topology change. Finish with the same bounded all-row additive color projection. This tests
  whether the interior loss is donor-shaped geometry debt rather than an unavoidable row-allocation
  cost; no iteration/LR/objective retuning is permitted.
- Stage-6 terminal closure (frozen after Stage 5 improved boundary/interior/detail metrics but
  ordinary-row motion opened 221 previously covered sites): apply the already frozen merge-funded
  micro operator to the Stage-5 endpoint for at most four waves, with no further geometry fit or
  changed selector. This is the terminal composition; it must keep the Stage-5 detail recovery and
  meet the original zero-hole/interior guard or the overall method is rejected.
- Stage-7 coverage-constrained recovery (frozen after Stage 6 closed holes but spent 222 additional
  micro rows and missed the interior guard): repeat Stage 5 from the corrected 873-micro base with
  the maintained boundary under-coverage hinge set to the existing boundary-phase values
  (weight=0.05, band=4 px, tau=0.05, cadence=8). All 221 Stage-5 holes were at SDF<=2.24, so no
  interior hinge or new threshold is introduced. Apply the same all-row terminal color projection;
  do not append a terminal closure unless reported as a separate arm.
- Stage-8 deep-only recovery (frozen after the maintained hinge reduced but did not eliminate new
  holes): repeat Stage 5 with no hinge, but train only ordinary rows whose centre is inside the
  already frozen fine-detail domain SDF>0.75+6.0=6.75 px. Freeze all boundary ordinary rows and
  micro rows; keep the 500-step objective and terminal projection unchanged. This is the final
  non-tuned topology-preserving recovery control.
- Stage-9 deep-only terminal closure (frozen after Stage 8 improved interior/detail metrics but
  opened 37 boundary sites through the long supports of moving deep rows): apply the unchanged
  Stage-6 merge-funded terminal closure to the Stage-8 endpoint, with no further fit, threshold,
  donor-score, or projection change. This confirmatory composition is the last method arm; no
  further outcome-driven method development is permitted in HIER-031.
- Primary coverage metrics: raw zero-coverage pixels, coverage<0.05 hole fractions in the <=4 px
  boundary and >4 px interior, connected hole components and largest component, SDF-binned holes,
  and coverage on mask skeleton/thin regions. No boundary-hole regression is permitted.
- Scaling-readiness diagnostic: partition the foreground spatially and report residual SSE,
  Gaussian density, local image complexity, and a common one-row proposal's estimated marginal
  gain. The target is not equal raw error per pixel; it is the absence of starved regions and
  approximately equal marginal value of the next row after accounting for local complexity.
- Quality metrics: foreground, <=4 px boundary, and >4 px interior PSNR; deep high-pass and
  Laplacian MSE; LPIPS when available; pixel maximum and 7x7 maximum. Report count, work, time,
  memory, acceptance/rejection telemetry, and exact-containment receipts.
- Promotion needs exact containment and count, zero raw coverage holes on the geometrically
  reachable foreground, no increase in unreachable holes, improved boundary/skeleton coverage,
  and no worse deep-interior PSNR beyond 0.05 dB. Appearance gains are secondary to topology.
- Produce a portable `index.html` with target/reconstruction/error/coverage/placement views,
  fixed hair/boundary/detail crops, a metric table, feasibility explanation, execution errors,
  hashes, and explicit diagnostic limitations. Do not overwrite HIER-030.

## Non-goals

- No Gaussian-count increase, 57.6k/native-resolution run, default flip, public-quality claim,
  endpoint mask payload, post-render masking, or relaxation of exact outside-zero containment.
- No claim that a silhouette skeleton alone identifies every photographic hair strand; mask
  topology and image high-frequency residual are scored separately.

## Acceptance criteria

- [x] The feasibility audit quantifies which pixels can and cannot be covered under current bounds.
- [x] At least the current exact-7k pipeline and boundary-recycle arms run to valid endpoints.
- [x] Any new allocation or micro-scale operator has focused determinism/containment tests.
- [x] All completed endpoints pass exact count/payload/containment/parity receipts.
- [x] A checked portable report contains the frozen metrics, visual comparisons, and errors ledger.
- [x] Producer review, docs/ARA synchronization, focused tests, and `verify.sh` complete.

## Interfaces touched

An experimental fitter hook preserves an independently certified, frozen micro-row cohort through
the ordinary constraint; it rejects dynamic topology and active-prefix storage. The HIER-031
driver/report/checker, focused tests, ADR-0033, architecture/research docs, and ARA evidence are
added. No CLI, endpoint schema, renderer semantic, persisted mask payload, or maintained default
changes.

## Depends on

HIER-030/029/028, CORE-010/011/012, FIT-023/025/040, ADR-0017/0019/0022/0025/0028/0030

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest
  `34afcdcf29b56adcb457e5838e2f2cc40efff0398725dbedee5b8b1ac6ea0d98`

### Outcome

The ordinary 0.35-pixel floor is a genuine representability blocker, not merely low count. The
mask has 980 pixels outside isotropic ordinary reach and, more decisively, ten pixels in three
connected components with no legal ordinary centre. A 0.08-pixel micro row has a 0.99-pixel
certificate radius and is legal at every active mask site.

The selected exact-N7,000 endpoint freezes 910 certified micro rows, optimizes only deep ordinary
rows, and funds its final 37-site closure with the unchanged count-neutral merge operator. It has
zero raw holes, zero raw thin-ridge holes, and exactly zero support/reconstruction outside. Against
the HIER-030 cold control it gains `+2.2844 dB` overall, `+2.3560 dB` in the boundary band, and
`+0.7546 dB` in the interior; high-pass MSE falls 6.31% and LPIPS 10.45%, while Laplacian MSE
worsens 4.56%. It is the only endpoint to pass the frozen topology/interior guard.

The untouched fixed pipeline is sharper (`25.2175 dB`, LPIPS `0.07828`) but leaves 933 raw holes;
capacity-time boundary recycling leaves 955. Visual review confirms broken hair/fringe holes in
those controls and connected but softer thin structures in the selected endpoint. The result
supports a topology-reserve mechanism on this exposed raster, not a default or generalization
claim. Raw-error equality is rejected as an allocation objective; marginal value is the relevant
quantity, and the reported proxy does not become more equal in this run.

Evidence:
`ara/evidence/hier031-exact7k-masked-boundary-detail-2026-08-12/run.md` and
`results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_2026-08-12/index.html`.
Interpretation note: `docs/research/2026-08-12-exact7k-masked-boundary-detail.md`.

### Handoff

#### Objective

Determine whether exact-N7,000 reallocation can eliminate masked holes and improve thin/fine
detail without scaling the count or leaking any support outside the foreground.

#### Changes

Added the source-bound feasibility audit, thirteen-endpoint sequential diagnostic, certified
fixed-micro fitter hook, fail-closed topology validation, report schema/checker support, focused
tests, ADR-0033, and synchronized task/docs/ARA records. Maintained defaults remain unchanged.

#### Evidence

The checker accepts the finalized 378-file report under explicit dirty/error-cell allowances.
Its presentation-finalization ledger proves that the black-matted objective-view repair changed no
field, metric, decision, feasibility record, or attempt. Every scored
endpoint has exactly 7,000 four-array rows and passes centre/support/reconstruction containment.
The selected endpoint passes the frozen zero-hole/interior guard; native-size source,
reconstruction, hair crop, error, coverage, placement, and hole views were inspected.

#### Assumptions

The repository max-side-1200 Janelle raster is the requested working resolution; the mask is known
at encoding time; three-sigma C0 support and a 0.75-pixel margin define exact containment.

#### Uncertainties

One exposed image/seed/device, dirty executed sources, sequential tuning on C0001, mixed detail
metrics, large additive coefficients, deterministic next-row proxy rather than a true marginal
oracle, and no native/density/rate/downstream or distinct-review evidence.

#### Review focus

Check the count-independent feasibility claim, exemption/freeze/topology invariants, exact
four-array/outside-zero receipts, selection guard, error-cell preservation, hair/hole visuals, and
whether the softness/conditioning caveats prevent overclaiming.

#### Protected actions not taken

No 57.6k/native run, no change to the ordinary global scale floor, no post-render masking, no
endpoint mask/cap payload, no threshold retune after Stage 9, no result overwrite, and no
method/default promotion.

#### Recommended next action

Obtain distinct review, then freeze disjoint mask/image confirmation. When count scaling is later
tested, retain a certified topology reserve and allocate new ordinary rows by measured marginal
gain rather than attempting to equalize raw error.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

Exact count/payload/containment receipts pass for all scored rows. The report checker accepts the
manifest and raw/table/link agreement. The fitter refuses every configuration that could change
the identity of an exempt row, and focused tests cover certificate preservation and validation.
The complete repository gate passes 1,992 tests with 26 skips and every structural checker green.

#### Evidence quality

The bundle is complete, portable, hash-bound, and visually inspected, but remains a dirty-source
sequential development diagnostic on one exposed raster. The explicit failed attempt is preserved.
No independent reviewer or held-out confirmation exists.

#### Simplicity

The selected endpoint remains an ordinary four-array Gaussian field. The only core hook is opt-in,
requires frozen independently certified rows, and fails closed around topology. The method itself
stays in an experiment driver.

#### Missing cases

Disjoint images and mask topologies, seeds/devices, native 5328x4608, approximately 57.6k density
parity, true marginal-gain measurement, coefficient conditioning controls, equal bytes/rate,
downstream use, and distinct code/scientific/visual review.

#### Required changes

Keep `formal_claim_ready=false`; do not call the selected endpoint sharper than the current
pipeline, claim that all hair is recovered, equate zero raw holes with uniform coverage, or infer
that later count scaling automatically reduces every local error.

#### Optional improvements

Add held-out topology-first confirmation and an allocation oracle that measures actual marginal
loss reduction per candidate row. Treat coefficient conditioning and weak-coverage reduction as
separate gates.
