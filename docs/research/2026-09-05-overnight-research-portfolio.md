# Research Portfolio: simpler-first StructSplat improvement

Repository/domain: finite-support 2D Gaussian image approximation.
Literature cutoff: 2026-09-05; searches on arXiv, author project pages, CVF, VLDB, publisher and author archives.
Provenance: user authorized overnight ideation, implementation and experiments; all candidates below are ai-suggested, not user-endorsed findings.
Status: prospective portfolio; experiment results belong in immutable bundles and task/ARA receipts, not in these hypotheses.

## 1. Frontier map

The maintained recipe uses a normalized renderer; the recent HIER track studies explicitly separate
additive fixed-count fields. Structural-tensor/WSE initialization, hierarchy, normalized coverage,
exact-count exchange, guarded RGB solves and alternating geometry already exist. HIER-031 exposed
terminal repair remains development evidence; HIER-032's negative coverage-debt result is a reason
to distinguish local repair from interior damage, not a license to tune a replacement on its outputs.

Dominant objects are Gaussian rows, per-pixel residuals and bounded edit transactions. Continuous
parameters coexist with discontinuous finite support and discrete row budgets. Shared assumptions
include residual-driven allocation, smooth local improvement, and rowwise summaries being adequate.
The unresolved mechanisms are gradient cancellation, ill-conditioned overlap, unreachable support,
recovery-dependent edit rankings and repeated geometry work during fixed-color-basis solves.

Recent frontier: LM-RS uses matrix-free second-order operators and residual sampling; 3DGS-LM
already caches sparse pixel/Gaussian derivative structure; 3DGS²-TR uses diagonal curvature and
parameterwise trust regions. Faster-GS separates numerical/implementation choices from algorithmic
changes. LocoADC couples local density control with consistency; TurboGS combines sparse supervision,
error-guided density and a curvature-informed optimizer. These make generic “second order,”
“cache it,” “local densification” or “sparse gradients” poor novelty claims.

## 2. Functional problem signature

Input is a bounded three-channel sampled signal and optional observation mask. Infer a finite set
of localized, overlapping basis functions and amplitudes. Allocate a fixed row budget, transport
its influence in the plane, and reduce global and worst-local errors within compute/memory limits.
RGB fitting is linear at fixed geometry; location/shape fitting is nonlinear; support membership
and count edits are discrete. Rotation, row permutation and some coefficient directions create
non-identifiability. Observed residual alone does not reveal whether capacity, geometry, appearance,
conditioning or support reachability limits progress. Measurements are development-only unless a
separate held-out protocol explicitly says otherwise. Hardware is a single RTX3050/8GiB GPU.

## 3. Fixation anti-library

Do not mistake more Gaussians, another perceptual loss, Adam tuning, a learned router, generic
attention, vague multiscale structure, residual-only births, or a CUDA rewrite for an explanation.
Keep normalized and additive hypotheses separate. Complexity earns its place only after a simpler
control fails. Independent generation lanes were kept separate before this audit; no global novelty
or publication claim is made. N3 labels below denote candidate formulation changes awaiting proof
and a wider prior-art search, not established inventions.

## 4. Productive recombinations

### Candidate A1 — Specialize the fixed-geometry color operator

- Central falsifiable claim: Caching weights should save repeated geometry work when enough PCG calls amortize construction.
- Novelty class / audit: N1 / likely known.
- Known foundation: 3DGS-LM sparse Jacobian caching; compiler specialization; existing HIER-010 PCG.
- Irreducible delta / A+B test: A bounded opt-in implementation with a complete-call, transaction-parity assay. This is deliberately A+B, not a novelty claim.
- Changed grammar or preserved transfer mechanism: Static support and geometry become an owned sparse linear map; coefficient values remain dynamic.
- New prediction: Acceleration should grow with operator reuse, but CSR construction can lose on small fields.
- Cheapest killing test / null: HIER-034: three backends, identical PCG, six counterbalanced repeats, build and peak memory included; kill workload-specific interchangeability on any parity/selection failure.
- Prior-art threats; success / partial / informative failure: CSR reduction ordering, memory pressure, and already-optimized streaming. A speed gain is an engineering result; partial gains identify break-even workloads; failure closes caching for this workload.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### Candidate A2 — Exact diagonal local curvature

- Central falsifiable claim: Scaling continuous updates by exact additive Gauss–Newton diagonal can improve convergence against a reasonably tuned Adam envelope.
- Novelty class / audit: N1 / likely known.
- Known foundation: 3DGS²-TR, LM-RS, 3DGS-LM; diagonal preconditioning is established.
- Irreducible delta / A+B test: Use the unusually transparent 2D additive Jacobian as a cheap exact diagnostic, not Hutchinson estimation. It is A+B.
- Changed grammar or preserved transfer mechanism: Keep primitives/count fixed; change only local update scaling and include backtracking cost.
- New prediction: Scale/color imbalance should matter more than well-conditioned isolated translations.
- Cheapest killing test / null: HIER-035: deterministic same-start fixtures, three Adam learning rates, diagonal and block controls; compare curves in steps and measured seconds. Kill if any benefit vanishes against the best Adam control.
- Prior-art threats; success / partial / informative failure: Known second-order splatting and implementation overhead. Success is a useful local optimizer; partial result identifies conditioning regimes; failure says Adam is sufficient here.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### Candidate A3 — Operator-aware gradient packets

- Central falsifiable claim: A signed/absolute/curvature packet can separate move, scale, color and split opportunities better than absolute position gradients alone.
- Novelty class / audit: N2 candidate / known components, relationship unconfirmed.
- Known foundation: AbsGS gradient collision, splitting steepest descent, repository Aug12 gradient anatomy.
- Irreducible delta / A+B test: Rank the action family, not just the Gaussian. The relationship may be useful; component combination is not transformational.
- Changed grammar or preserved transfer mechanism: Preserve pixel contributions before row reduction and contrast continuous cancellation against split curvature.
- New prediction: Symmetric width errors can have near-zero translation gradient without requiring extra capacity.
- Cheapest killing test / null: HIER-033: finite enumerated edits, exact trial evaluation, equal recovery budget and count accounting; kill a predictor if its ranking regret is not better than simpler controls.
- Prior-art threats; success / partial / informative failure: AbsGS, GDAGS, SteepGS and LocoADC substantially overlap. Success supports a bounded action diagnostic; partial result narrows identifiable families; failure documents ambiguity.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### Candidate A4 — Alternating exact colors and cheap geometry

- Central falsifiable claim: A few geometry steps between color projections may recover more quality per second than solving either block too accurately.
- Novelty class / audit: N1 / likely known.
- Known foundation: Variable projection; HIER-014 alternating refinement; nonlinear least squares.
- Irreducible delta / A+B test: A stopping schedule based on measured marginal benefit. Existing alternation is the baseline, not an invention.
- Changed grammar or preserved transfer mechanism: Geometry changes invalidate cached color operators; rebuilding is paid in full.
- New prediction: Over-solving colors early should be wasteful when geometry immediately changes.
- Cheapest killing test / null: Compare fixed short/long alternating schedules against existing refinement at equal measured time; no schedule search on held-out sources.
- Prior-art threats; success / partial / informative failure: FIT-046/HIER-014 and variable projection. Success is a scheduling improvement; partial gains define regimes; failure discourages more orchestration.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

## 5. Exploratory candidates

### B1 — Replace mandatory densification by an action auction

- Central falsifiable claim: Many high-residual rows need a continuous edit rather than an added primitive.
- Novelty class / audit: N2 candidate / known relationship.
- Known foundation: A3, LocoADC, existing residual exchange.
- Irreducible delta / A+B test: Reverse the assumption that gradient pressure means insufficient count; make edit type an output. This remains an exploratory combination.
- Changed grammar or preserved transfer mechanism: Every proposed split competes against move/scale/color and a no-op under an explicit budget.
- New prediction: At fixed N, action-aware edits should beat split-only rules on width/translation cases.
- Cheapest killing test / null: Small oracle bank first; reject if simple residual ranking with the same recovery budget matches it.
- Prior-art threats; success / partial / informative failure: Best-of-many trial bias and tuned action magnitudes. Success is a routing rule; partial result gives counterexamples; failure favors a simpler policy.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### B2 — Support transitions are events, not smooth gradients

- Central falsifiable claim: Pixels outside finite support create an observable blind spot that local differential information cannot resolve.
- Novelty class / audit: N2 candidate / insufficient evidence.
- Known foundation: Compact-support optimization; HIER-032 coverage failures.
- Irreducible delta / A+B test: Treat first support contact as a discrete event whose cost is evaluated explicitly; generic event-driven optimization threatens novelty.
- Changed grammar or preserved transfer mechanism: State includes support-entry events, rather than assuming all residuals are differentiably reachable.
- New prediction: An uncovered target can have zero gradients for every current row despite nonzero residual.
- Cheapest killing test / null: Translate one atom toward an uncovered lobe; compare gradient-only, finite support-event and residual-birth controls. Kill if a simple fixed-step search resolves the same cases at equal cost.
- Prior-art threats; success / partial / informative failure: Boundary discontinuities and arbitrary event radii. Success characterizes blind spots; partial success localizes them; failure shows no new control is needed.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### B3 — Coefficient equivalence classes instead of individual rows

- Central falsifiable claim: Near-duplicate atoms create unidentifiable color directions whose removal can improve conditioning without visible loss.
- Novelty class / audit: N2 candidate / likely known foundation.
- Known foundation: Rank-revealing least squares, quotient spaces, merging methods.
- Irreducible delta / A+B test: Use renderer-equivalent coefficient directions as the diagnostic object; compression novelty is not established.
- Changed grammar or preserved transfer mechanism: Optimize a local observable subspace, then map back to the unchanged representation.
- New prediction: Strongly correlated support pairs should have large coefficient drift but small rendered drift.
- Cheapest killing test / null: Construct two gradually coincident atoms; measure singular values, coefficient norm, rendered error and solve time. Kill if ridge alone matches the result.
- Prior-art threats; success / partial / informative failure: Standard regularization and merging may explain everything. Success gives a conditioning diagnostic; partial success flags unsafe counts; failure retains ridge.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### B4 — Local stopping rather than globally uniform iterations

- Central falsifiable claim: Regions with stationary residual could stop updating while difficult regions receive the saved compute.
- Novelty class / audit: N2 candidate / known components.
- Known foundation: TurboGS sparse supervision; asynchronous block-coordinate optimization.
- Irreducible delta / A+B test: Use a conservative freeze/reactivate certificate with whole-image checks. Sparse updating alone is known.
- Changed grammar or preserved transfer mechanism: Change optimization allocation, not the Gaussian representation.
- New prediction: Useful speedups require error persistence and weak cross-region overlap.
- Cheapest killing test / null: Two coupled regions with one easy component; compare uniform updates, static sparse updates and reactivation, charging checks. Kill on missed reactivation or no wall-time gain.
- Prior-art threats; success / partial / informative failure: Overlap couples regions and stale residuals can lie. Success is bounded compute allocation; partial result locates coupling thresholds; failure favors uniform work.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

## 6. Transformational candidates

### Candidate C1 — Representability certificates before optimization

- Central falsifiable claim: A cheaply computed attainable-image bound can distinguish an impossible fixed-support error target from a poorly optimized one.
- Novelty class / audit: N3 candidate, NOT established / insufficient evidence.
- Known foundation: Convex dual certificates, interval bounds, finite-support geometry; E4 below.
- Irreducible delta / A+B test: Output a witness about the admissible representation before proposing new primitives. The changed object is an attainable-error certificate, not another loss.
- Changed grammar or preserved transfer mechanism: For fixed geometry and coefficient bounds, seek a lower bound on achievable residual; geometry uncertainty makes it conservative.
- New prediction: Some coverage failures should be certified before spending optimizer iterations.
- Cheapest killing test / null: Tiny exhaustive fixed-basis problems: compare certificate bounds with exact constrained solves. Null: bound is vacuous. Abandon if useful bounds require solving the original problem or falsely exclude feasible images.
- Prior-art threats; success / partial / informative failure: Convex feasibility theory is a major threat; Gaussian-specific formulation novelty is unverified. Success could explain impossibility; partial result certifies only holes; failure is a useful limit on certification.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### Candidate C2 — A topology edit as a budget-conserving transport plan

- Central falsifiable claim: Jointly pricing where capacity leaves and arrives can outperform independently choosing a donor and a birth site.
- Novelty class / audit: N3 candidate, NOT established / insufficient evidence.
- Known foundation: Optimal transport/resource allocation; existing exact-count exchange.
- Irreducible delta / A+B test: Primitive proposal becomes a coupled donor–recipient plan with recovery cost, not two independent scores. This may collapse to established matching.
- Changed grammar or preserved transfer mechanism: Conserve active-row budget; include all donor damage in the finite trial.
- New prediction: The best recipient alone should often not form the best funded edit.
- Cheapest killing test / null: Exhaustive 4-donor by 4-recipient analytic bank, compare separable ranking with joint gains. Null: gains approximately factor. Abandon if separable matching is sufficient.
- Prior-art threats; success / partial / informative failure: HIER-011 residual exchange and auction methods; no surveyed global novelty. Success exposes nonseparability; partial success finds local cases; failure supports simple donor rules.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### Candidate C3 — Fit a local image-change operator before choosing primitives

- Central falsifiable claim: A low-rank local desired image change may transfer across equivalent Gaussian parameterizations more reliably than rowwise scores.
- Novelty class / audit: N3 candidate, NOT established / insufficient evidence.
- Known foundation: Tangent-space optimization, reduced-order models, natural gradients.
- Irreducible delta / A+B test: Make an image-space intervention the proposal primitive, then compile it into admissible Gaussian edits. This changes proposal grammar, but may reduce to Gauss–Newton.
- Changed grammar or preserved transfer mechanism: Separate desired observable change from its non-unique parameter realization.
- New prediction: Duplicate or rotated-equivalent parameterizations should lead to equivalent chosen image edits.
- Cheapest killing test / null: Two equivalent fields and a tiny enumerated action bank; compare image-space and rowwise selection under identical candidates. Null: no invariance advantage. Kill if solving the image-space step costs as much as exhaustive trials.
- Prior-art threats; success / partial / informative failure: Natural-gradient and variable-projection formulations threaten the entire delta. Success could yield an invariant selector; partial success gives a diagnostic; failure rejects added abstraction.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

## 7. Cross-domain transfers

The fourth primitive-generation proposal, C4 (an implicit decoder program replacing Gaussian
rows), was rejected before selection: subtracting generic implicit representations left no
bounded recipient-specific prediction, and decoder/rate accounting would expand the overnight
scope. The three C candidates above survive only as unverified hypotheses.

For each transfer, "Barrier" denotes its adoption barrier and "Broken" its broken correspondence.
The central claim is falsifiable; every killing test states a null hypothesis or an explicit
failure signature. These labels do not upgrade a speculative transfer's novelty.

### E1 — Compiler partial evaluation → fixed-support specialization

- Central falsifiable claim: Specializing static geometry avoids recomputing invariant weights in dynamic RGB solves.
- Novelty class / audit: N1-T / likely known.
- Known foundation: Futamura partial evaluation; 3DGS-LM caching.
- Irreducible delta / A+B test: Map program+static inputs → renderer+geometry/mask; residual program → sparse RGB operator. This is systems reuse, not a new formulation.
- Changed grammar or preserved transfer mechanism: Preserved mechanism: precompute invariants. Broken: geometry changes and floating reduction order. Invention: invalidation boundary. Barrier: build memory; enabling change: long fixed-geometry PCG blocks.
- New prediction: Only enough operator reuse can amortize build cost.
- Cheapest killing test / null: HIER-034 compares complete calls, not kernel microbenchmarks; parity and memory are mandatory.
- Prior-art threats; success / partial / informative failure: Native baseline is unchanged streaming. Terminology removal leaves caching, so downgrade to N1. Success/partial/failure are break-even findings.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### E2 — Database incremental views → sparse cache updates

- Central falsifiable claim: When only a few Gaussians move, updating affected sparse columns can be cheaper than rebuilding the entire operator.
- Novelty class / audit: N2-T / known components, bridge unconfirmed.
- Known foundation: DBToaster incremental view maintenance; sparse matrix updates.
- Irreducible delta / A+B test: Map database rows → Gaussian geometry rows, materialized view → pixel incidence matrix, delta query → changed-support update.
- Changed grammar or preserved transfer mechanism: Preserved: exact localized recomputation. Broken: dynamic support changes cardinality and sorted storage. Required: affected-pixel invalidation. Barrier: CSR rebuilds; enabling change: retained scatter chunks.
- New prediction: Benefit should depend on changed-support fraction, not just moved-row count.
- Cheapest killing test / null: Move 1/5/25/100 percent of atoms; compare full rebuild with exact deltas and assert matrix/adjoint equality. Kill if bookkeeping dominates at realistic fractions.
- Prior-art threats; success / partial / informative failure: Native full rebuild and standard sparse updates are strong threats. Success enables local refinement; partial finds crossover; failure rejects incremental complexity.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### E3 — Experimental design → an action-discriminating atlas

- Central falsifiable claim: Choosing synthetic cases where competing predictors disagree can reveal selector failures using fewer expensive trials.
- Novelty class / audit: N2-T / measurement transfer, rarely connected.
- Known foundation: Atkinson–Fedorov discrimination design; robust discriminating experiments.
- Irreducible delta / A+B test: Map rival response models → rival edit predictors; controlled stimuli → analytic images; discriminating design → cases that separate action rankings.
- Changed grammar or preserved transfer mechanism: Preserved: choose interventions to discriminate mechanisms. Broken: predictors can overfit the atlas. Required: freeze families and perturbations before outcomes. Barrier: synthetic validity; enabling: exact renderer oracles.
- New prediction: Width mismatch and two-lobe mismatch should separate hypotheses that agree on raw residual magnitude.
- Cheapest killing test / null: HIER-033: freeze a small causal bank; use additional unseen perturbations only in a later protocol. Null: atlas does not change rankings or hypotheses.
- Prior-art threats; success / partial / informative failure: Native random/smooth/edge fixtures are baseline. This creates evidence, not a quality method. Success is a discriminating benchmark; partial isolates one ambiguity; failure limits the observable.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### E4 — Abstract interpretation → conservative support certificates

- Central falsifiable claim: A conservative abstraction of attainable support/color can certify some impossible requests without full fitting.
- Novelty class / audit: N3-T candidate, NOT established / rare formulation transfer.
- Known foundation: Cousot–Cousot abstract interpretation; C1.
- Irreducible delta / A+B test: Map concrete image states → Gaussian parameter sets; abstract domain → pixel support/color intervals; sound transfer → bounds on rendered contribution.
- Changed grammar or preserved transfer mechanism: Preserved: conservative over-approximation excludes only truly impossible targets. Broken: correlations make intervals loose. Required: a useful correlated abstract domain. Barrier: vacuous bounds; enabling: finite support and bounded RGB.
- New prediction: Thin disconnected components may admit useful certificates while dense interiors remain inconclusive.
- Cheapest killing test / null: Exhaustive tiny systems verify soundness; then compare certificate strength and cost against direct optimization. Kill on one false infeasibility witness or consistently vacuous bounds.
- Prior-art threats; success / partial / informative failure: Native geometric hole checks/convex duality may subsume it. Success changes the output to a witness; partial only diagnoses holes; failure rejects the grammar change.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### E5 — Numerical nuisance elimination → geometry after exact colors

- Central falsifiable claim: Eliminating easy linear coefficients can expose the harder geometry objective with less optimizer interference.
- Novelty class / audit: N1-T / likely known.
- Known foundation: Variable projection / nuisance-parameter estimation; HIER-014.
- Irreducible delta / A+B test: Map nuisance parameters → RGB coefficients; profiled objective → best bounded color fit at each geometry.
- Changed grammar or preserved transfer mechanism: Preserved: analytically/numerically remove an inner variable block. Broken: bounded transaction selection is not a smooth exact argmin. Required: distinguish actual selected solve from ideal profiled derivative. Barrier: inner cost; enabling: cheap color cache.
- New prediction: Inner accuracy should have a measurable diminishing return.
- Cheapest killing test / null: Compare inner tolerances and geometry steps at equal wall time. Kill if cheap alternating Adam is as good.
- Prior-art threats; success / partial / informative failure: Existing variable projection dominates novelty. Success/partial/failure concern useful inner-solve accuracy, not a new optimizer theory.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### E6 — Reliability fault injection → report and solver falsification

- Central falsifiable claim: A report that rejects intentionally corrupted controls gives more trustworthy negative and positive decisions.
- Novelty class / audit: N1-T / measurement transfer, rarely connected.
- Known foundation: Fault-injection testing, metamorphic software tests.
- Irreducible delta / A+B test: Map injected failures → mismatched geometry, incomplete cells, source drift, wrong metrics; fault detector → protocol gates.
- Changed grammar or preserved transfer mechanism: Preserved: test that the safety mechanism fails closed. Broken: finite injected faults cannot prove validity. Required: scientific failure classes, not only syntax corruption. Barrier: extra fixtures; enabling: immutable per-cell artifacts.
- New prediction: Internally rehashed corruption should still fail semantic identity checks.
- Cheapest killing test / null: HIER-034 report tests inject matrix/digest/trace/selection/SSE faults. Kill the reporting implementation if any known violation produces an eligible positive verdict.
- Prior-art threats; success / partial / informative failure: Native schema checks are baseline. This is validation engineering; success closes demonstrated gaps, partial identifies unchecked classes, failure blocks claims.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

## 8. New-evidence discovery programs

### D1 — Finite-edit oracle atlas

- Central falsifiable claim: A small causal intervention bank can determine which gradient observables actually predict useful edit families.
- Novelty class / audit: N2 evidence / relationship unconfirmed.
- Known foundation: E3 and A3.
- Irreducible delta / A+B test: Vary defect type, amplitude, anisotropy and seed; measure prediction rank, immediate gain, recovery gain and trial cost. Evidence, not another renderer.
- Changed grammar or preserved transfer mechanism: Freeze bank and oracle candidates independently of outcomes; derive gradients and check autograd/finite differences.
- New prediction: A selector can be locally correct yet fail after recovery, changing the hypothesis from score quality to recovery interaction.
- Cheapest killing test / null: HIER-033 analytic action trials; missing actions and count-changing edits must be explicitly labelled rather than hidden.
- Prior-art threats; success / partial / informative failure: Exclude support-boundary derivative artifacts and best-of-many budgets. Success yields a causal map; partial gives isolated mechanisms; failure kills an overgeneral selector.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### D2 — Color-solve cost anatomy

- Central falsifiable claim: Build cost, reduction order and checkpoint overhead can explain apparent backend speedups.
- Novelty class / audit: N1 evidence / established measurement practice.
- Known foundation: E1/E6; Faster-GS methodology.
- Irreducible delta / A+B test: Vary overlap, mask, scale, N and backend; measure complete time, cache time, memory and every checkpoint.
- Changed grammar or preserved transfer mechanism: Compare identical initial states and full iteration traces; do not call repeated runs independent images.
- New prediction: A kernel speedup may fail complete-call or transaction parity, inducing a different optimization target.
- Cheapest killing test / null: HIER-034 frozen matrix and two cache arms; no retuned threshold after results.
- Prior-art threats; success / partial / informative failure: Exclude warmup/compilation imbalance and incomplete repeats. Success identifies useful workloads; partial finds thresholds; failure redirects work away from caching.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

### D3 — Conditioning versus coverage phase diagram

- Central falsifiable claim: Optimization failure changes character between ill-conditioned overlap and unreachable support.
- Novelty class / audit: N2 evidence / insufficient evidence.
- Known foundation: B2/B3/C1.
- Irreducible delta / A+B test: Vary atom separation, scale, target distance and coefficient bounds; measure Gram spectrum, attainable residual, gradient packet and optimizer curves.
- Changed grammar or preserved transfer mechanism: Separate representation impossibility from optimizer failure using tiny exact controls.
- New prediction: A sharp transition from slow progress to zero useful gradient would suggest event-aware edits, not a stronger smooth optimizer.
- Cheapest killing test / null: Start with two atoms on a 32x32 analytic field, exhaustive target placements and bounded color solves. Kill explanatory claim if spectra/coverage do not separate observed failure modes.
- Prior-art threats; success / partial / informative failure: Renderer truncation and coefficients can confound the phase boundary. Success is a mechanism map; partial narrows one regime; failure rejects a convenient explanation.
- Novelty confidence: low unless marked likely known; bounded search to the cutoff above, sources below. Unsearched patent/thesis formulations remain an explicit limitation.

## 9. Pareto frontier

Scores are prospective judgments, 0–5, not measured outcomes. First-test cost is scored with 5
meaning cheapest; interpretation, baseline strength and negative-result value are distinct.

| Candidates | Novelty | Falsifiable | Explanatory | Importance | Feasible | Cheap test | Interpretable | Baselines | Negative value | Publication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1/E1/D2 | 1 | 5 | 3 | 2 | 5 | 5 | 5 | 5 | 4 | 1 |
| A2 | 1 | 5 | 4 | 3 | 5 | 4 | 5 | 5 | 4 | 2 |
| A3/D1/E3 | 2 | 5 | 5 | 4 | 4 | 4 | 5 | 4 | 5 | 3 |
| A4/E5 | 1 | 4 | 3 | 3 | 4 | 4 | 4 | 5 | 3 | 1 |
| B1 | 2 | 5 | 4 | 4 | 4 | 3 | 4 | 4 | 4 | 3 |
| B2 | 2 | 5 | 5 | 4 | 4 | 4 | 5 | 4 | 5 | 3 |
| B3 | 2 | 5 | 4 | 3 | 4 | 5 | 5 | 5 | 4 | 2 |
| B4 | 1 | 4 | 3 | 3 | 3 | 3 | 4 | 4 | 3 | 2 |
| C1/E4 | 3 | 5 | 5 | 4 | 2 | 4 | 4 | 4 | 5 | 4 |
| C2 | 3 | 5 | 4 | 4 | 3 | 4 | 4 | 5 | 4 | 3 |
| C3 | 3 | 4 | 5 | 4 | 2 | 3 | 3 | 5 | 4 | 4 |
| E2 | 2 | 5 | 3 | 3 | 3 | 3 | 5 | 5 | 4 | 2 |
| E6 | 0 | 5 | 2 | 3 | 5 | 5 | 5 | 5 | 5 | 1 |
| D3 | 2 | 5 | 5 | 4 | 4 | 4 | 5 | 5 | 5 | 3 |

The fastest validation/system direction is A1. A3/D1 is the strongest informative-negative and
action-diagnostic direction. C1/E4 is the strongest theoretical, high-risk direction but is not
an overnight implementation priority. A2 offers the simplest independent convergence control.

## 10. Recommended first experiment

Run HIER-034 after distinct prospective review: unchanged streaming against bounded scatter/CSR,
count and geometry fixed, build and all checkpoint work included. Null: neither cache achieves
1.1x median paired speedup with every integrity gate on a workload. No threshold rescue.
In parallel development (not overlapping timed GPU workloads), prepare the HIER-033 finite-edit
oracle and HIER-035 Adam/curvature controls. Implement these bounded tasks through the normal method,
benchmark, review and docs-sync workflow; the ideation skill itself does not authorize code changes.

## 11. Audit limitations and source trail

No publication potential score is a promise. This is a bounded primary-source search, not a
patent, thesis or exhaustive synonym review. Several old donor mechanisms are plainly established.
The cross-domain mappings are our inferences, not claims made by those authors. A transfer survives
terminology removal only when its exact map, failure correspondence and native baseline remain.
C1/C2/C3 fail an established-novelty claim for now: subtraction leaves a potentially useful
formulation, but conventional feasibility, matching and natural-gradient explanations remain.
No evidence here establishes a normalized-renderer, actual-rate, whole-pipeline or held-out gain.

Primary sources (concise paraphrases above; no borrowed numerical speed claims):

- [LM-RS project](https://vcai.mpi-inf.mpg.de/projects/LM-RS/): matrix-free second-order splat optimization and residual sampling.
- [3DGS-LM, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Hollein_3DGS-LM_Faster_Gaussian-Splatting_Optimization_with_Levenberg-Marquardt_ICCV_2025_paper.pdf): strong prior art for sparse cached derivative/operator work.
- [3DGS²-TR](https://arxiv.org/abs/2602.00395): diagonal Hessian approximation and trust regions.
- [Faster-GS](https://arxiv.org/abs/2602.09999): optimization and implementation analysis.
- [LocoADC](https://arxiv.org/abs/2607.17896): locality-aware density control.
- [TurboGS](https://arxiv.org/abs/2606.15924): sparse supervision, error-guided density and hybrid optimization; identified in the fresh search.
- [AbsGS](https://arxiv.org/abs/2404.10484): gradient-collision prior art, already in the repository's Aug12 research context.
- [Gradient-Direction-Aware Density Control](https://arxiv.org/abs/2508.09239): direction-aware density-control prior art, rechecked in the fresh search.
- [SteepGS, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.html): splitting matrices, saddle-point conditions and two-offspring directions are established prior art; our additive oracle is not a new splitting theorem.
- [Nuisance-parameter estimation](https://arxiv.org/abs/1206.6532): variable-projection foundation.
- [DBToaster](https://arxiv.org/abs/1207.0137): incremental materialized-view maintenance.
- [Futamura, partial evaluation](https://fi.ftmr.info/PE-Museum/PE-memo.PDF): specializing computation with static inputs.
- [Atkinson–Fedorov discriminating experiments](https://academic.oup.com/biomet/article-abstract/62/2/289/337065): experimental model discrimination.
- [Robust discriminating designs](https://arxiv.org/abs/1309.4652): robustness of competing-model designs.
- [Cousot–Cousot, abstract interpretation](https://www.di.ens.fr/~cousot/COUSOTpapers/POPL77.shtml): conservative abstract semantics and sound approximation.

## 12. Unrun follow-up sketch: direction rescue versus deeper backtracking

Status: ai-suggested design, independently critiqued by codex-overnight-protocol-reviewer;
not implemented, executable-approved, or run. HIER-036 is prior exposed evidence, not a
prospective confirmation set for this sketch. Create a new task and freeze an executable
digest before any future formal execution. This is a known optimization safeguard, not a
novel method claim or an explanation of the observed texture behavior.

- Question: after a full-coupling GN direction exhausts its trial budget, does changing
  direction help terminal quality more than merely extending that same backtracking sequence?
- Proposed matrix: the unchanged HIER-036 texture generator at64x64/N16; exposed conditions0–5
  and additional procedural conditions6–8, reported separately. Six arms: Adam multipliers
  .3/1/3, full_shared GN with6trials, GN with12trials, and GN6 followed by6projected-gradient
  rescue trials only after every GN trial rejects. All54cells required;160terminal attempts.
- Retain HIER-036 objective, renderer, bounds, damping and trust vector. Let D=diag(trust),
  s=maxabs(Dg). Rescue direction is -D²g/s if s>0, otherwise exactly zero. Every trial starts
  from the same unchanged pre-update state. GN12 tries factors2^0 through2^-11; the hybrid
  tries GN factors2^0 through2^-5, then restarts the gradient sequence at2^0 through2^-5.
  Finite non-increasing loss, including ties and zero movement, is acceptable; all-failure
  retains exact state. Reuse the existing gradient/Jacobian/solve for the fallback.
- Fairness: GN12 and the hybrid share a worst-case ceiling of12trial renders, not equal
  realized computation. Each curvature arm constructs160Jacobians and solves160systems;
  Adam performs160backward evaluations. Charge every render. Record GN/rescue trial counts,
  fallback activations, acceptance route, rejection, and g·actual_displacement after bounds
  projection for every proposal, including whether projection changed the proposal.
- Proposed separate gates: hybrid versus GN12 and hybrid versus strongest-Adam envelope,
  each per stratum, median terminal gain>=.5dB, no condition loss>.1dB, MS-SSIM loss<=.005,
  LPIPS increase<=.01, with whole-matrix integrity. Preference requires both comparisons in
  both strata. Hybrid versus GN6 stays descriptive. No speed or matched-compute claim.
- Required tests: all arms share their first six GN trials at the same state; rejection
  ownership and counters; zero gradient and completely blocked movement; a constrained
  quadratic where box-clipped GN loses descent but projected gradient can decrease loss.
  For example g=(1,1), H=[[5,3],[3,2]] gives unconstrained d=(1,-2); blocking negative motion
  in coordinate2 leaves (1,0). This generic example is not a diagnosis of the texture cases.
- Limits: this does not isolate the ultimate failure cause. A standalone projected-gradient
  arm would be required before a GN–gradient synergy claim. Natural-image transfer, dense
  scalability and practical perceptual relevance remain separate future questions.
