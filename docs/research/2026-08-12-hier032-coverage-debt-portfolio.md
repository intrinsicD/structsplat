# Research Portfolio: Exact-count masked coverage debt

**Repository/domain:** StructSplat, exact-count masked 2D additive Gaussian image fields.

**Literature cutoff:** 2026-08-12.

**Sources searched:** the live StructSplat task/ADR/ARA history; arXiv primary records for
[Revising Densification in Gaussian Splatting](https://arxiv.org/abs/2404.06109),
[AbsGS](https://arxiv.org/abs/2404.10484), and
[Coverage Axis](https://arxiv.org/abs/2110.00965); functional searches for pixel-error Gaussian
densification, absolute-gradient/detail densification, medial-ball set cover, masked Gaussian
coverage, fixed-count donor merging, and local least-squares merge error.

**Key unresolved assumptions:** one exposed raster represents other mask topologies; unit coverage
0.05 is a useful geometric floor beyond this endpoint; station-ball candidates remain useful at
other counts; and local additive merge error predicts global post-projection damage.

The portfolio is an adversarial record around the already selected HIER-032 killing experiment.
It does not authorize the deferred candidates below. The implemented relationship is classified
as **known components, possibly new relationship**, N1/N2 under this search, and carries no
global novelty claim.

## 1. Frontier map

HIER-031 established the immediate frontier. Its selected four-array N=7,000 field has zero raw
holes and exact outside-zero containment, yet 743 foreground pixels have unit coverage below 0.05
in 483 8-connected components. Of those, 461 lie in the frozen hair crop and the weak set accounts
for about 39% of foreground SSE. A deterministic prototype requires roughly 700 certified
placements, so most debt is isolated one-pixel capacity rather than a few broad missing regions.

| Work or component | Problem | Primitive | Mechanism | Evidence | Threat or gap |
|---|---|---|---|---|---|
| HIER-031 | raw masked holes at exact N | certified 0.08-px micro rows | merge-funded closure plus deep-only recovery | one exposed C0001 field | closes zero coverage, not weak coverage |
| Revising Densification | weak ADC allocation | per-pixel error | pixel-error-driven primitive generation with count control | multi-scene 3DGS evaluation | directly threatens any pixel-error novelty claim |
| AbsGS | blurred fine detail from gradient collision | absolute/homodirectional view-space gradient | cancellation-resistant densification signal | multi-dataset 3DGS evaluation | directly threatens any absolute-detail detector novelty claim |
| Coverage Axis | compact shape skeletonization | interior medial balls | global set cover over surface samples | 3D shape benchmarks | directly threatens medial/set-cover novelty |
| ADR-0019 station-ball cap | exact mask containment | tangent ellipse plus covering balls | certify support against the SDF | reference tests and HIER-030/031 | supplies candidate legality, not allocation optimality |
| HIER-031 donor merge | exact-count funding | mutual-nearest row pair | covariance envelope, recertificate, project RGB | exposed diagnostic | donor score is geometric/appearance heuristic, not exact damage |

Dominant approaches either add capacity where error is high, preserve thin geometry through medial
structure, or redistribute fixed capacity with a merge heuristic. The sparse region is their
interaction under a strict four-array exact-N endpoint, exact outside-zero support, and a hard
minimum coverage gate. The anomaly worth testing is that zero raw holes coexist with concentrated
weak-coverage error.

## 2. Functional problem signature

Input is a bounded binary domain, a target RGB signal, and a fixed collection of compact-support
basis functions. Hidden state is which pixels lack sufficient basis support and which existing
basis pairs can be compressed with least damage. The conserved quantity is row count. Coverage is
geometric and nonnegative; reconstruction is signed RGB and can cancel. Candidate placement is
discrete, coefficient projection is continuous, containment is a hard boundary condition, and GPU
summation adds small execution-order uncertainty. The identifiable question is whether moving
capacity from locally redundant pairs to certified debt sites improves boundary and hair quality
without violating the interior floor.

## 3. Fixation anti-library

The obvious suggestions are disallowed or weak controls here: increase N; lower the 0.05 gate;
dilate or relax the mask; black-mask the rendered output afterward; globally shrink the minimum
scale; rerun geometry optimization after seeing placements; use only residual magnitude; or add a
learned allocator trained on the exposed image. These either change the question, evade exact
containment, consume new capacity, or confound placement with later optimization.

## 4. Productive recombinations

### Candidate method P1 — Certified fallback closure

**Central claim:** one certified 0.08-px row at every weak pixel can close geometric debt at exact
N when funded by the existing HIER-031 donor operator.

**Novelty class:** N1.

**Known foundation:** HIER-031 micro closure and pixel-error densification.

**Irreducible delta:** change the trigger from zero coverage to a preregistered positive coverage
deficit.

**Why this is not merely A + B:** it is deliberately an A+B causal control, not a novelty claim.

**Changed grammar or transfer mechanism:** none; it stays in the existing four-array field grammar.

**New prediction:** exact closure will require nearly one donor pair per weak pixel and may damage
interior quality.

**Cheapest killing test:** the frozen fallback arm. **Null hypothesis:** donor-funded fallbacks fail
closure or breach the interior floor.

**Prior-art threats:** Revising Densification and HIER-031 cover the components.

**Novelty confidence:** 0–10% for mechanism novelty; cutoff 2026-08-12; primary papers and live
repository searched, patents and unpublished branches not exhaustively searched.

**Scientific value:** establishes whether the weak set is genuine capacity debt.

**Publishable if successful:** only as a control supporting a broader fixed-count study.

**Publishable if partially successful:** morphology of reopened debt identifies donor harm.

**Publishable if it fails informatively:** proves a positive coverage threshold is not closed by
point fallbacks under the current donor contract.

### Candidate method P2 — Component-wise certified set cover

**Central claim:** inward-offset, maximum certified tangent ellipses reduce the rows required to
satisfy every 0.05 deficit relative to per-pixel fallbacks.

**Novelty class:** N1/N2.

**Known foundation:** Coverage Axis set cover, ADR-0019 station-ball containment, and pixel-error
densification.

**Irreducible delta:** sparse incidence is defined over *remaining positive coverage deficit* in a
fixed-count Gaussian image field, with appearance variance as a lower-priority tie-break.

**Why this is not merely A + B:** the components are recognizable; only the exact-N deficit
incidence and donor-coupled gate remain after subtraction. That remainder is treated as a
possibly new relationship, not a new primitive.

**Changed grammar or transfer mechanism:** Coverage Axis transfers global coverage selection from
surface samples/medial balls to weak image pixels/certified tangent ellipses. The preserved causal
mechanism is compact global coverage; the broken correspondence is that Gaussian weights are soft,
deficits vary, and RGB error matters.

**New prediction:** compression will be modest because most components are singletons, but a small
number of tangent rows will cover multiple hair/boundary pixels.

**Cheapest killing test:** compare selected count and closure against P1. **Null hypothesis:** the
set cover selects no fewer rows or fails exact closure.

**Prior-art threats:** Coverage Axis is the strongest; facility-location and weighted set-cover
work may already express the same abstraction.

**Novelty confidence:** 10–30% for the relationship, 0–5% for components; cutoff and search scope as
above.

**Scientific value:** tests whether component geometry can compress isolated debt.

**Publishable if successful:** as part of a broader fixed-rate allocation mechanism with disjoint
evidence.

**Publishable if partially successful:** separates compressible boundary chains from singleton
capacity demands.

**Publishable if it fails informatively:** quantifies the lower bound imposed by isolated pixels.

### Candidate method P3 — Contribution-aware merge funding

**Central claim:** ranking recertified mutual-nearest merges by exact local additive SSE after a
local RGB least-squares fit preserves more boundary, hair, and interior quality than the HIER-031
distance/color/scale/angle score for the same placement policy.

**Novelty class:** N2.

**Known foundation:** mutual-nearest contraction, covariance-envelope merging, variable
projection, and local reconstruction-error ranking.

**Irreducible delta:** evaluate the actual local additive reconstruction equation after
recertification before choosing disjoint donor pairs.

**Why this is not merely A + B:** it remains a recognizable direct-control refinement; the useful
claim is causal superiority of the exact local criterion, not novelty.

**Changed grammar or transfer mechanism:** donor selection changes from parameter similarity to
predicted functional damage while the representation stays fixed.

**New prediction:** the donor-error distribution will separate pairs that look similar in
parameter space but differ in local cancellation context.

**Cheapest killing test:** frozen arms P2 and P3 with identical first-wave placement. **Null
hypothesis:** P3 does not improve both hair and boundary quality or fails closure.

**Prior-art threats:** error-aware pruning/merging and local least-squares basis reduction are
likely known under other names.

**Novelty confidence:** 10–25% for the exact task relationship; incomplete patent/thesis search.

**Scientific value:** distinguishes allocation damage from placement value.

**Publishable if successful:** only after broader fixed-count, multi-image confirmation.

**Publishable if partially successful:** donor-error telemetry can explain metric tradeoffs.

**Publishable if it fails informatively:** parameter similarity is sufficient at this scale.

## 5. Exploratory candidates

### Exploratory candidate E1 — Boundary high-pass strokes after closure

**Central claim:** a fixed 128-row batch selected by absolute high-pass residual, 2-px NMS, SDF≤4,
and image tangent improves hair and boundary quality without reopening coverage debt.

**Novelty class:** N1.

**Known foundation:** AbsGS-style cancellation resistance, FIT-040 high-pass pursuit, and
ADR-0019 tangent support.

**Irreducible delta:** apply detail only after geometric closure and fund it with the same exact
local donor score.

**Why this is not merely A + B:** it is an explicit A+B arm intended to test complementarity.

**Changed grammar or transfer mechanism:** absolute detail detection is transferred from gradient
densification to a frozen boundary residual batch; geometry remains unchanged afterward.

**New prediction:** detail can improve while coverage remains closed, but interior quality may
fall because an additional 128 donors are consumed.

**Cheapest killing test:** the fifth HIER-032 arm. **Null hypothesis:** it fails either hair,
boundary, closure, or the interior floor.

**Prior-art threats:** AbsGS and FIT-040 directly cover the detector family.

**Novelty confidence:** 0–15%; no novelty claim.

**Scientific value:** tests complementarity between geometry debt and appearance debt.

**Publishable if successful:** as a stage-order result in a larger confirmed method.

**Publishable if partially successful:** identifies which metric the fixed detail batch trades.

**Publishable if it fails informatively:** shows closure capacity and detail capacity compete.

### Exploratory candidate E2 — Deficit dual prices

Treat each weak pixel's unsatisfied mass as a dual price and select placements/donors jointly.
Novelty class N2/N3 candidate; defer because it changes the frozen selector and needs a new task.

### Exploratory candidate E3 — Component morphology strata

Predeclare singleton, chain, junction, and blob strata, then report closure and quality by stratum.
Novelty class N1 measurement candidate; useful after HIER-032 but not an outcome-dependent rescue.

## 6. Transformational candidates

### Transformational candidate T1 — Coverage debt as a first-class field

Replace binary birth events with a conserved, continuously transported debt measure paired with
row-capacity credits. Novelty class N3 candidate. The new primitive would be debt transport rather
than Gaussian addition. Its new prediction is path-independent closure cost under equivalent
allocation sequences. Cheapest killing test: find two allocation orders with different terminal
credit/debt balance. Prior-art threat: primal-dual facility location.

### Transformational candidate T2 — Certificate-native representation grammar

Make a support certificate, rather than an unconstrained covariance, the encoded primitive; RS
parameters become a decoded realization. Novelty class N3 candidate. New prediction: every decoded
field satisfies containment by construction across optimization and quantization. Cheapest killing
test: construct a codec-rounding counterexample. Prior-art threat: constrained shape primitives and
safe-by-construction codecs.

### Transformational candidate T3 — Functional donor equivalence classes

Define rows by equivalence of their local rendered column space rather than parameter proximity.
Novelty class N3 candidate. New prediction: functionally equivalent donor classes admit bounded
merge loss independent of RS distance. Cheapest killing test: search for close-parameter/high-loss
and far-parameter/low-loss counterexamples. Prior-art threat: reduced-basis and column subset
selection literature.

## 7. Cross-domain transfers

### Transfer X1 — Medial set cover from computational geometry

Preserved mechanism: globally cover samples with few interior primitives. Broken correspondence:
surface membership is binary, while Gaussian deficit satisfaction is weighted. Required invention:
sparse weighted incidence under containment. Adoption barrier: certificate cost and singleton-heavy
components. Novelty class N2-T at most.

### Transfer X2 — Reliability reserve from power systems

Treat weak components as local reliability deficits and donor pairs as reserve funding. Preserved
mechanism: capacity must remain feasible after reallocating reserve. Broken correspondence: there is
no network flow law or N-1 contingency. Required invention: a Gaussian functional-damage receipt.
Adoption barrier: the analogy may add no predictive value. Novelty class N2-T measurement transfer.

### Transfer X3 — Column subset selection from numerical linear algebra

Treat Gaussian rows as rendering columns and donor merging as local basis compression. Preserved
mechanism: choose a reduced basis by functional approximation error. Broken correspondence: columns
depend on clipped anisotropic geometry and global coefficient bounds. Required invention: cheap
local exact error with a global projection guard. Adoption barrier: full leverage-score machinery is
too expensive. Novelty class N2-T.

### Transfer X4 — Deficit clearing from market design

Weak pixels bid with deficit mass; donor merges supply row credits. Preserved mechanism: scarce
capacity is allocated by marginal value. Broken correspondence: bids are not independent and RGB
coefficients couple globally. Required invention: truthful or monotone marginal estimates. Adoption
barrier: the market vocabulary may obscure a simpler primal-dual optimizer. Novelty class N3-T only
if the formulation yields order-independent clearing; otherwise N1 analogy.

## 8. New-evidence discovery programs

### Evidence program D1 — Capacity-demand morphology

Vary mask topology and count while measuring weak-component size, certified cover compression,
and placements-to-closure. A surprising stable singleton law would motivate a capacity lower-bound
hypothesis. Exclude bugs with fallback completeness, exact coverage rerenders, and synthetic masks.

### Evidence program D2 — Donor-score calibration

For every eligible pair, record parameter score, local fitted merge SSE, global post-projection
damage, and regional metrics. A rank reversal that repeats across images would support functional
donor selection. Exclude leakage by freezing pairs before outcomes and using disjoint images.

## 9. Pareto frontier

| Candidate | Apparent novelty | Falsifiability | Importance | Feasibility | First-test cost | Informative failure | Publication potential |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 fallback | 1 | 5 | 3 | 5 | 2 | 5 | 1 |
| P2 set cover | 2 | 5 | 3 | 5 | 2 | 5 | 2 |
| P3 donor error | 2 | 5 | 4 | 4 | 3 | 5 | 3 |
| E1 high-pass | 1 | 5 | 3 | 5 | 2 | 5 | 2 |
| T1 debt field | 4 | 3 | 4 | 2 | 4 | 3 | 4 |
| T2 certificate grammar | 4 | 4 | 4 | 2 | 4 | 4 | 4 |
| D2 calibration | 2 | 5 | 4 | 4 | 3 | 5 | 3 |

## 10. Recommended first experiment

Run HIER-032's frozen five-arm exact-N7,000 development protocol. **Null hypothesis:** no successor
arm simultaneously closes all raw and <0.05 coverage holes, preserves outside-zero support and
reconstruction, improves both boundary and hair-crop quality, and keeps interior PSNR at or above
35.2631 dB. The decisive observation is an arm passing every clause; cost is one exposed
single-image CUDA run. Abandon the selected relationship if no arm passes, and retain the best
tradeoff only as negative diagnostic evidence.

## 11. Adversarial prior-art audit

The strongest reconstruction of HIER-032 from prior art is: use Revising Densification's pixel
error to identify underrepresented pixels, AbsGS/FIT-040 to avoid cancellation in the optional
detail detector, Coverage Axis to compress coverage candidates, and standard local least squares
to rank donor merges. That combination leaves no defensible component-level novelty. The only
irreducible remainder found is the exact-N, exact-containment coupling between positive unit-
coverage debt, certified tangent incidence, and donor-error funding. It is therefore labeled
"known components, possibly new relationship," not apparently transformational.

| Prior work | Problem overlap | Representation overlap | Mechanism overlap | Prediction overlap | Evidence overlap | Overlap threat |
|---|---:|---:|---:|---:|---:|---|
| Revising Densification | high | medium | high | medium | different 3D setting | high |
| AbsGS | medium | medium | high for E1 | medium | different 3D setting | high for detector |
| Coverage Axis | high for cover | low/medium | high | medium | different shape setting | high for selector |
| HIER-031 | high | exact | high | high | same exposed source | highest implementation threat |

Search confidence is deliberately low-to-moderate. Primary papers and the repository were
accessible; patents, theses, paywalled indexes, unpublished code branches, and every historical
term for error-aware basis reduction were not exhaustively searched. The probability that the
relationship has an unlocated close analogue is estimated at 40–70%. This range is qualitative,
not a statistical posterior.

## 12. Audit limitations

The live repository context supersedes the ideation skill's dated profile. Search breadth is not a
systematic review. The exposed HIER-031 field shaped the question, so HIER-032 remains development
evidence even with a clean commit and prospective protocol review. No result here can support a
default, general compression, native-resolution, actual-rate, or broad novelty claim.
