# Research Portfolio: code-driven convergence and evaluation cost

Post-run disposition: the [completed findings](2026-09-05-code-driven-findings.md) record
FIT-050/PORT-007 plus the separately frozen FIT-051 follow-up, all214 cells and independent
audits. Every utility/promotion gate is negative; small backtracking and component timing
observations remain bounded. Preserve the prospective designs below as design history, not
as current unrun status. Other portfolio proposals remain untested; no default or novelty claim.

Repository: StructSplat, normalized constant-color 2D fields. Cutoff: 2026-09-05.
Scope: hypotheses and experiment designs, not outcome or novelty claims. Two bounded tasks
are selected: FIT-050 and PORT-007. No new renderer equation, default, external compute purchase,
sealed-data access or publication claim is authorized by this portfolio.

## 1. Frontier map

| Code / prior evidence | Primitive, assumption and operator | Open question |
|---|---|---|
| fit._solve_colors_normalized | Fixed geometry gives linear normalized RGB; unpreconditioned CG solves an absolute-color system | Does a smaller guarded displacement give useful safe progress when its endpoint does not? |
| safe_schedule._safe_color_solve | Whole proposal receives the complete Pareto gate; rollback owns parameters and optimizer state | Is useful progress discarded between parent and endpoint? Existing event-CG negative evidence is not permission to retime the same policy. |
| safe_schedule.evaluate_quality | CUDA RGB then separate torch raw-coverage pass; top-k tail then full quantile sort | Can same-call intermediates be reused without changing discrete decisions? |
| fit / _MaskConstraint.apply | Adam moments persist through projected geometry and topology edits | Are rejected normal components of momentum repeatedly reintroduced? |
| safe_schedule / _safe_fit_block | Global tail statistics and coverage constrain locally supported edits | Support disjointness alone does not imply independent safe transactions. |
| HIER-033–036, FIT-011/027, BENCH-018 | Existing operator, cache, curvature, moment and gate studies | Do not relabel their negative or narrow results as a new practical improvement. INDEX and ARA retain their exact dispositions. |

The dominant paradigm is parameter-space optimization followed by global measurement and
transactional acceptance. Geometry is nonlinear, colors linear at fixed geometry, topology
discrete, and CUDA sums nondeterministic. Strongly explored: timers, loss swaps, coarse-to-fine,
local curvature, generic residual splitting and persistent basis caching. Less measured:
repeated full quality-evaluation work; safe fractions of rejected endpoints; numerical
scale dependence; moment/constraint contact. Sparse is not synonymous with novel.

## 2. Functional problem signature

Input: bounded samples of a signal and a possibly constrained domain. Hidden state:
locations, anisotropic footprints and signed coefficients. Allocate a finite number of local
contributors, normalize their overlap, and improve a global set of error and feasibility
statistics. Continuous updates coexist with discrete membership changes. A local update may
change globally ranked errors; near-boundary floating-point changes can alter acceptance.
Color translation is linear only while geometry, opacity, support and normalization remain
fixed. Limits are basis expressivity, ill-conditioning, measurement cost and safe-step rejection.

## 3. Fixation anti-library

Do not lead with more Gaussians, attention, a learned scheduler, another loss, more hierarchy,
a timer tweak, a bigger dense Hessian, or an unpriced cache. These can be controls but are not
transformations by name. We prefer a one-dimensional transaction and elimination of duplicated
work before constructing a new optimizer. Novelty, usefulness, correctness and speed are
separate axes.

## 4. Productive recombinations — independent lane A

Each card's central claim is a hypothesis. For these N1/N2 cards, the A+B test usually fails:
the value sought is an implementation or causal finding, not an irreducibly new algorithm.

### Idea A1 — Safe normalized color ray (selected FIT-050)

- Central claim: a gradient/Jacobi direction with guarded fractions can yield useful safe
  image improvement at less complete work than the existing CG endpoint.
- Novelty class: N1; known foundation: steepest descent, Jacobi scaling, exact quadratic line
  minimization, backtracking and transaction rollback.
- Irreducible delta: recipient-specific comparison against CG endpoint, CG interpolation and
  inherited-moment Adam under the unchanged full protected-metric gate. It is deliberately A+B.
- Mechanism: fix the normalized basis A, form g=Aᵀ(target-Ac), choose v=g or
  g/(diag(AᵀA)+lambda), test one scalar ray, and re-render before committing.
- New prediction: an endpoint can fail while a nonzero interior fraction passes; this says
  nothing yet about beating Adam or converging repeatedly.
- Cheapest killing test: the frozen four-image/two-seed one-transaction assay. Abandon the
  utility claim if the minimum gain or any no-harm guard fails; rollback is not success.
- Prior-art threats: classical CG/line search, 3DGS-LM, LM-RS and visibility-weighted color
  least squares. Novelty confidence: high confidence in known ingredients; no global novelty
  probability is asserted under the limited sources below.
- Scientific value: success supports a bounded practical refinement; partial success explains
  guard/endpoint mismatch; failure separates basis limits from unsafe-step policy. None alone
  warrants a paper or a default change.

### Idea A2 — Same-call measurement reuse (selected PORT-007)

- Central claim: shared RGB/raw coverage and shared upper-tail order statistics reduce full
  evaluator cost without altering gates.
- Novelty class: N1; known foundation: common-subexpression elimination and order statistics.
- Irreducible delta: exact normalized denominator ownership, conservative fallbacks and
  all-reasons/A-A/full-pipeline controls. This is useful A+B, not a new renderer or estimator.
- Mechanism: retain forward's already-produced denominator; derive p99 from the sorted tail
  already needed by CVaR, preserving input-dtype rank and interpolation arithmetic.
- New prediction: gains depend on evaluator's cost share and fallback frequency, not merely
  on kernel throughput.
- Cheapest killing test: same-state complete evaluations followed by a bounded full pipeline;
  abandon execution-equivalence if discrete decisions diverge, even with close final PSNR.
- Prior-art threats: auxiliary outputs in gsplat, compiler CSE, database materialization,
  PyTorch quantile implementation. Novelty confidence: known engineering, no novelty claim.
- Value: success supports measured opt-in acceleration; partial success identifies a component
  bottleneck; failure maps an equivalence boundary. Publication would require broader evidence.

### Idea A3 — Residual-centered, scale-aware CG

- Central claim: solving a correction rather than absolute colors removes avoidable
  cancellation/stopping sensitivity. Novelty class: N1.
- Known foundation: iterative refinement and preconditioned Krylov methods.
- Irreducible delta / grammar: no new grammar; isolate centering, preconditioning and relative
  stopping as three factors, not an unidentifiable bundled solver replacement.
- New prediction: intensity rescaling changes the old breakdown location more than the
  correction solver's relative residual.
- Cheapest killing test: fixed tiny full-rank and rank-deficient bases at several amplitudes,
  checked against float64 dense solutions. Stop if the difference is only a faulty oracle.
- Prior-art threats: classical numerical linear algebra; novelty confidence: likely known.
- Value: a numerical regression suite on success, a precision-bound diagnosis on partial
  success, or a clean rejection if no practically sized field benefits. Deferred.

### Idea A4 — Footprint-coordinate geometry updates

- Central claim: optimizing means in a frozen local footprint frame reduces anisotropy-induced
  step imbalance. Novelty class: N1/N2.
- Known foundation: coordinate preconditioning and natural-gradient geometry.
- Irreducible delta: Gaussian support-frame parameterization frozen per block; multiplying an
  Adam gradient alone is not sufficient because its moment normalization cancels scaling.
- New prediction: improvement correlates with footprint aspect ratio, not image identity.
- Cheapest killing test: elongated versus isotropic fixtures with strongest global-LR and
  isotropic-coordinate controls, equal evaluations. Kill if global LR explains the gain.
- Prior-art threats: reparameterized optimizers, LM-RS; novelty confidence: low, not searched
  exhaustively. Success/partial/failure value: conditioner, mechanism boundary, or ruled-out
  explanation, respectively. Deferred.

## 5. Exploratory candidates — independent lane B, assumption surgery

### Candidate B1 — Projected-momentum contact

- Central claim: repeatedly discarding outward parameter motion while retaining its Adam
  moment wastes updates at active mask/scale constraints. Novelty class: N2.
- Known foundation: tangent-cone projected optimization. Irreducible delta: change only the
  outward moment component at observed contact; do not zero all moments.
- Changed assumption: projection is not independent of optimizer state.
- New prediction: contact-heavy masked cases benefit; full-frame/no-contact controls do not.
- Cheapest killing test: contact census, then frozen unchanged/zero-all/selective controls.
  Kill if contact is rare or selective loses to simple lower LR.
- Prior-art threats: projected Adam and active-set optimizers; novelty confidence: insufficient
  bridge audit. Value across success/partial/failure: targeted fix, contact diagnostic, or
  abandoned state hypothesis. Existing moment-tempering results remain controls.

### Candidate B2 — Endpoint feasibility is an interval, not one bit

- Central claim: some rejected CG endpoints contain a useful feasible substep. Novelty class: N1.
- Known foundation: line search; irreducible delta: none beyond the recipient's multi-metric
  transaction. Changed grammar: endpoint choice becomes a six-point ordered feasible search.
- New prediction / killing test: CG interpolation versus its own charged endpoint in FIT-050;
  zero accepted nontrivial fractions rejects the mechanism on this slice.
- Prior-art threats: all guarded line-search methods; novelty confidence: known. Success
  clarifies step feasibility, partial success estimates its frequency, failure rejects this
  explanation. This is an arm of A1, not an independent experiment or extra discovery count.

### Candidate B3 — Row age rather than global Adam age

- Central claim: exact row-local bias correction helps newly born rows more reliably than
  broad moment tempering. Novelty class: N1/N2.
- Known foundation: Adam bias correction and dynamic parameter sets. Irreducible delta:
  distinguish birth age from scheduler time; changed assumption is a single optimizer clock.
- New prediction: differences are concentrated immediately after birth, not on old rows.
- Cheapest killing test: a fixed scheduled birth with inherited/zero/local-age controls;
  kill if matching first-step magnitude explains all benefit.
- Prior-art threats: dynamic embedding optimizers and existing FIT-011; novelty confidence:
  insufficient, low priority. Success/partial/failure yield a bounded state rule, a step-size
  explanation, or a reason not to revisit this crowded direction.

### Candidate B4 — Commuting local color transactions

- Central claim: locally supported color changes can sometimes be composed with fewer global
  measurements. Novelty class: N2-T candidate.
- Known foundation: conflict-serializable transactions. Irreducible delta: account explicitly
  for normalization and globally ranked protected errors.
- Changed assumption: global validation after each edit might be replaced by a proved
  composition condition; support disjointness alone is not that condition.
- New prediction: sparse overlap plus stable tail membership helps, dense fields do not.
- Cheapest killing test: compose two individually safe edits and search for a CVaR violation;
  kill the simple rule on the first counterexample.
- Prior-art threats: graph coloring, block coordinate descent, transactions; novelty
  confidence: low. Success gives a sufficient condition, partial success narrows it,
  failure is a useful counterexample. Deferred until such a condition exists.

## 6. Transformational candidates — lane C, proposed rather than established novelty

These introduce formal objects absent from the frontier above. They survive as questions,
not validated transformations. Subtraction/necessity tests may downgrade all of them.

### Candidate C1 — Proof-carrying image delta

- Central claim: a local field update can carry a compact certificate sufficient to validate
  protected metrics more cheaply than re-rendering the complete image.
- Novelty class: proposed N3-T, currently insufficient evidence.
- Known foundation: proof-carrying code, interval arithmetic and incremental computation.
- Irreducible delta: a delta plus an error/coverage certificate becomes the admissible object,
  rather than an unconstrained field proposal. If it is merely cached sums, downgrade to N1.
- New prediction: certificate size scales with affected support and rank crossings.
- Cheapest killing test: two overlapping Gaussian edits with adversarial near-tau pixels and
  tied tail ranks; derive conservative bounds and compare certificate work to full evaluation.
  Null hypothesis: the certificate is too loose or expensive to avoid full evaluation.
- Prior-art threats: certified incremental computation and sufficient statistics. Novelty
  confidence: low under this limited search. Success could support a theorem and mechanism;
  partial success a restricted exact update; informative failure a certificate-size lower bound.

### Candidate C2 — Admissible image-change cone

- Central claim: optimize a feasible image displacement first, then lift it to coefficients,
  to avoid repeatedly proposing directions that the safe gate forbids.
- Novelty class: proposed N3-T, not an established novelty claim.
- Known foundation: multiobjective feasible directions, control barriers and constrained
  least squares. Irreducible delta: define K from foreground/boundary/tail directional
  constraints and unchanged support, then solve for d in K intersect range(A).
- Changed grammar: the primitive is an admissible image motion, not a parameter gradient.
  It fails subtraction if ordinary multiobjective projection gives the identical operation.
- New prediction: infeasible gradient mass, rather than residual size alone, predicts stalls.
- Cheapest killing test: a tiny dense basis and exhaustive active-set oracle; compare projected
  gradient/penalty/line search at equal work. Kill if the cone is trivial or the known oracle
  already implements exactly the proposed formulation.
- Prior-art threats: Exact Pareto Optimal Search and convex multiobjective line search.
  Novelty confidence: low. Success could establish a feasibility characterization; partial
  success a stall diagnostic; failure an equivalence or impossibility result.

### Candidate C3 — Tail-interference hypergraph

- Central claim: a graph of protected-metric interactions predicts when individually safe
  changes can be composed, more accurately than geometric overlap alone.
- Novelty class: proposed N3-T; known foundation: conflict graphs and hypergraph scheduling.
- Irreducible delta: nodes are candidate image deltas; hyperedges encode joint tail-rank or
  coverage violations. If pairwise geometry explains everything, downgrade to N2.
- Changed grammar: transactions and their admissible compositions replace a list of splats.
- New prediction: disjoint supports can still share a tail-budget hyperedge.
- Cheapest killing test: enumerate pairs/triples of tiny basis updates against the exact
  global gate; compare support graph, pairwise graph and hypergraph prediction. Kill if
  constructing the hypergraph costs at least as much as serial validation.
- Prior-art threats: distributed transactions, higher-order constraint satisfaction, block
  coordinate optimization. Novelty confidence: low; success/partial/failure yield a
  composition rule, a restricted predictor, or a concrete nonlocality counterexample.

### Candidate C4 — Owned measurement packet (downgraded to A2)

- Central claim: treating RGB, raw denominator and derived tail statistics as one same-call
  value removes redundant work with no cross-call invalidation protocol.
- Novelty class: N1 after the grammar/subtraction tests.
- Known foundation / prior-art threats: tuples, auxiliary renderer outputs and CSE.
- Irreducible delta: none theoretically; ownership is a useful API contract.
- New prediction and cheapest killing test: PORT-007's same-state and pipeline assays.
- Novelty confidence: confidently ordinary engineering. Success/partial/failure are useful
  cost/parity findings, not a transformational publication.

## 7. Cross-domain transfers — lane E

Six donor mechanisms, not six metaphors. Database, reliability, economics and operations
research are deliberately distant from the Gaussian optimizer; compilers and control are
closer. All predictions are untested here.

| Transfer / donor → recipient | Preserved mechanism and required invention | Broken correspondences (at least three) | Adoption barrier, enabling change and native baseline |
|---|---|---|---|
| E1 databases: materialized shared expression → RGB/coverage packet (A2, N1-T) | Compute once, reuse within one transaction; expose raw normalized denominator | atomics are not exact relations; geometry changes invalidate data; tiny threshold errors change predicates | Avoid retained cache lifecycle by same-call ownership; compare ordinary repeated evaluation, including overhead |
| E2 compilers: available expression → shared tail selection (A2, N1-T) | One order-statistic computation serves two consumers; match rank dtype and lerp | floating associativity; NaN/Inf ordering; large-N wrapper changes estimator | Pin arithmetic and fall back outside support; compare pinned torch quantile and topk separately |
| E3 control: feasibility barrier → safe color fraction (A1/C2, N2-T or proposed N3-T) | Restrict motion to admissible directions before committing | discrete holes; nonsmooth tail ranks; renderer roundoff | Exact color linearity and actual replay enable a bounded test; compare ordinary line search and projected gradient |
| E4 reliability: redundant measurement near decision boundary → gray-zone fallback (A2, N2-T diagnostic) | Use reference verification when numerical uncertainty can flip classification | no universal CUDA error bound; correlated sums; fallback has a cost | Same-field exact hole masks and A/A expose failures; compare always-reference and always-reuse, never claim a proof from a tolerance |
| E5 economics: Pareto-admissible exchange → image-change cone (C2, proposed N3-T) | Seek improvements with no protected party worse off; lift feasible motion into a realizable basis | metrics are not utilities; finite steps not derivatives; attainable image range is restricted | Tiny active-set oracle makes feasibility observable; compare established multiobjective projected directions |
| E6 operations research: conflict scheduling → tail-interference graph (B4/C3, N2-T/proposed N3-T) | Compose operations only under explicit incompatibility constraints | global ranking; dynamic supports; higher-order conflicts may defeat pairwise graphs | First derive a sufficient condition, then schedule; compare serial equal-work transactions and support-only coloring |

Terminology removal leaves identifiable operations in every row. E1/E2 are historically obvious
and downgraded. E3/E5 share mathematics and cannot be counted as independent algorithmic
discoveries. E4 transfers a measurement practice, not a faster renderer. For E6 the simplest
counter-analogy is two disjoint edits that change tail membership globally; that is a required
counterexample search, not rhetoric.

## 8. New-evidence discovery programs — lane D

### Program D1 — Measurement-cost and decision-equivalence atlas (selected PORT-007)

Vary coverage reuse and tail reuse independently on identical initial/terminal fields, then
complete full-frame/masked pipelines. Measure full evaluation time, fallback, every scalar,
pixel classifications, decision reasons and A/A variation. A large component gain with no
pipeline gain identifies Amdahl's limit; decision divergence despite image parity exposes
discontinuous control sensitivity. Bind source/input hashes, retain every repetition/error,
counterbalance order and deny isolated timing claims under foreign compute.

### Program D2 — Numerical scale metamorphisms (deferred)

Vary signal amplitude, basis conditioning and ridge scale while holding the exact normalized
operator fixed. Measure breakdown, true residual, dense float64 error and complete work.
Unexpected amplitude sensitivity suggests absolute stopping/cancellation rather than a better
basis is needed. Check algebra and derivatives independently; keep singular and zero cases,
never choose a favorable amplitude after outcomes.

### Program D3 — Constraint-contact census (deferred)

Vary full-frame versus controlled masks and record projection displacement, outward moment,
time since birth and next-step rejection. Contact concentrated in a small row population would
motivate B1; absent contact kills it cheaply. Log before changing policy, preserve no-contact
controls and separate numerical clipping from actual mask contact. No adaptive selection on
a confirmation split.

## 9. Pareto frontier and independent value audit

Scores are subjective pre-experiment 0–5 priorities, not probabilities or evidence.
Cost score is affordability (5 cheapest). X explanatory value; I importance; R interpretable
result; B strong baselines; F informative failure; P publication potential.

| Candidate | Novelty | Falsifiable | X | I | Feasible | Cost | R | B | F | P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 / B2 | 1 | 5 | 3 | 3 | 5 | 5 | 5 | 5 | 4 | 1 |
| A2 / C4 / D1 | 1 | 5 | 4 | 3 | 5 | 5 | 5 | 5 | 5 | 1 |
| A3 / D2 | 1 | 5 | 4 | 3 | 5 | 5 | 5 | 5 | 5 | 2 |
| A4 | 2 | 4 | 3 | 3 | 4 | 4 | 3 | 4 | 3 | 2 |
| B1 / D3 | 2 | 4 | 4 | 3 | 4 | 4 | 4 | 4 | 4 | 2 |
| B3 | 1 | 4 | 3 | 2 | 4 | 4 | 4 | 4 | 4 | 1 |
| B4 / C3 | 3 | 4 | 4 | 3 | 2 | 3 | 4 | 4 | 5 | 3 |
| C1 | 3 | 4 | 5 | 4 | 2 | 3 | 4 | 3 | 5 | 3 |
| C2 | 3 | 5 | 5 | 4 | 3 | 4 | 5 | 5 | 5 | 3 |

Fastest validation and systems value: A2. Cheapest practical quality test: A1.
Strongest potential theory/high-risk direction: C1/C2. Best negative-result value: D2 and
C3's nonlocality counterexample. None is selected by summing these scores.

## 10. Recommended first experiment

Run A1 and A2 through FIT-050 and PORT-007, respectively. Their task files, executable PROTOCOL
objects and source digests are the sole frozen experiment authorities. Null hypotheses:
no useful safe color gain, and no complete-cost reduction under preserved decisions.
Do not enlarge the method or dataset to rescue either null result. Full objective, budget,
primary-arm choice, effect/no-harm gates and exact commands are frozen before formal outcomes.
Simple implementation is the priority; the three formulation questions remain unimplemented.

## 11. Prior-art audit limitations

Independent code generation and prospective review preceded execution. Reviewer comparison
already downgrades safe rays to classical optimization and renderer auxiliary outputs to
engineering. Search families include color normal equations, Gaussian LM/PCG, feasible
multiobjective directions, shared subexpressions and exact quantile arithmetic.

Primary sources consulted:
[CG tutorial](https://people.eecs.berkeley.edu/~jrs/papers/cg.pdf),
[3DGS-LM](https://lukashoel.github.io/3DGS-LM/),
[LM-RS](https://vcai.mpi-inf.mpg.de/projects/LM-RS/),
[Instant Colorization](https://arxiv.org/abs/2604.17155),
[gsplat renderer API](https://docs.gsplat.studio/main/apis/rasterization.html),
[PyTorch 2.9 quantile source](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/aten/src/ATen/native/Sorting.cpp),
[Exact Pareto Optimal Search](https://proceedings.mlr.press/v119/mahapatra20a/mahapatra20a.pdf),
[convex multiobjective line search](https://arxiv.org/abs/2404.10993),
[shared-subexpression materialization](https://www.microsoft.com/en-us/research/publication/selecting-subexpressions-to-materialize-at-datacenter-scale/).

These sources threaten the named facets; none establishes global novelty or absence of prior
work. Patent/thesis/older-terminology and distant-field coverage is incomplete; source
availability and recency vary. No absolute novelty probability can be calibrated from this
search. C1–C3 are proposed grammar changes pending deeper audit, not apparently established
transformations. Successful tests could still yield only useful engineering, and that is an
acceptable outcome for the user's request.

Independent analytic audit (not an executed experiment) further narrows C1–C3. For 101 pixel
losses with nonzeros (0.10,0.09,0,0.08,0.08), CVaR99 is the mean of the two largest entries.
Disjoint edits A: first two to (0.11,0), B: last three to (0.09,0,0) each lower total error
without increasing CVaR or p99. Together they raise the top-two sum from 0.19 to 0.20.
Individual safety plus disjoint supports therefore does not prove joint safety. This is an
algebraic counterexample to the naive composition rule, not evidence that a Gaussian field
realizes every specified loss vector. A certificate must establish joint admissibility.
Likewise, an unrestricted admissible image direction can become unsafe after projection
onto range(A); C2 must optimize the intersection from the outset. At tail ties it must handle
all active tail sets. These threats make a standard constrained-optimization interpretation
more plausible than an irreducibly new formulation.

The independent reviewer ultimately downgrades C1–C3 to N1/N2-T pending a demonstrated
recipient-specific theorem or cost advantage. Additional nearest threats are
[proof-carrying code](https://people.eecs.berkeley.edu/~necula/pcc.html),
[self-adjusting computation](https://www.cs.cmu.edu/~guyb/papers/ABBT06.pdf),
[multiobjective steepest descent](https://link.springer.com/article/10.1007/s001860000043),
[invariant confluence](https://www.vldb.org/pvldb/vol8/p185-bailis.pdf), and the
[CVaR epigraph formulation](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf).
The generated transformational lane is retained for traceability, not defended after audit.
