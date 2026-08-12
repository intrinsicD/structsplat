# Research Portfolio: Gradient-informed topology control for HIER and 2D-field-to-3DGS lift

**Repository/domain:** StructSplat HIER fields and their downstream use as persistent 3D Gaussian
initializers in `realtime-gs`.

**Literature cutoff:** 2026-08-12.

**Sources searched:** the live StructSplat HIER/FIT/task/ARA history; the sibling `realtime-gs`
density controller, topology seam, roadmap, and evidence ledger; primary records or official code
for [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[Revising Densification](https://arxiv.org/abs/2404.06109),
[AbsGS](https://arxiv.org/abs/2404.10484),
[3DGS as MCMC](https://arxiv.org/abs/2404.09591),
[SteepGS](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.html),
[GDAGS](https://arxiv.org/abs/2508.09239),
[LocoADC](https://arxiv.org/abs/2607.17896), and
[REFINE](https://arxiv.org/abs/2606.09074); and functional searches around function-preserving
growth, matching pursuit, influence-based pruning, and split/merge state-space search.

**Status:** research proposal only. It changes no default and authorizes no outcome-dependent
implementation. The recommended next step is a frozen operator-identifiability oracle, not a new
production controller.

**Key unresolved assumptions:** a source 2D field's error decomposition predicts useful downstream
3D topology; short-horizon counterfactual gains predict recovered gains; action probes can be made
cheap enough for real-time-oriented training; and persistent field lineage is sufficiently stable
after projection, occlusion, and optimizer state migration.

## Executive answer

Gradient information can materially improve HIER density control, but **the ordinary aggregated
parameter gradient cannot by itself choose among clone, split, merge, prune, and teleport**.
It is best understood as a measurement of local sensitivity inside the current parameterization.
Topology edits change that parameterization.

The useful decomposition is:

1. **source-field stress:** where a fitted 2D field is under-representing its source image;
2. **downstream 3D stress:** which persistent 3D Gaussian receives coherent evidence across fitted
   views;
3. **operator value:** whether one particular discrete, count-balanced rewrite lowers the actual
   downstream objective after its unavoidable render perturbation and short recovery.

The proposed relationship is a **balanced counterfactual transaction auction**. For each legal
topology transaction, compute or approximate its rendered perturbation, score that perturbation
against the current image-space loss gradient, locally refit appearance, and exact-render only the
best few proposals. First-order gradients can rank many finite clone/merge/prune/teleport proposals;
a symmetric split needs a second-order splitting matrix or a finite nonzero probe. Source 2D scores
are proposal priors only: the final decision must be made in the persistent multi-view 3D field.

## 1. Frontier map

| Work or local component | Signal | Operator rule | Useful foundation | Remaining gap |
|---|---|---|---|---|
| Original 3DGS ADC | accumulated view-space mean-gradient norm | scale threshold chooses clone or split; opacity/size prunes | simple, fast, widely implemented | magnitude says stress, not which edit has value |
| AbsGS | sum of absolute per-pixel positional subgradients | densify high-score large rows | exposes gradient cancellation | discards directional coherence and does not unify operators |
| Revising Densification | auxiliary per-pixel reconstruction error assigned to contributors | error-driven additions with count control | sees high error when parameter gradient is blind | contributor error is not a discrete action value |
| GDAGS | gradient coherence ratio | low coherence favors split; high coherence favors clone | directly answers split-versus-clone with gradient direction | strongly occupies that mechanism; not prune/merge/teleport |
| SteepGS | per-Gaussian splitting matrix | split only for a negative eigenvalue and use its eigenvector | principled second-order split test | only one edit family and added kernel complexity |
| 3DGS as MCMC | opacity/redundancy and relocation sampling | relocate dead rows to live sites | fixed-capacity transport alternative | destination value and initializer interaction still need matched tests |
| LocoADC | region-wise 2D image error/coherence and local similarity | densify regions and merge similar rows | direct 2D Gaussian allocation and merging prior art | source-image success need not imply 3D lift value |
| REFINE | rendering-aware approximate Hessian field | predict removal importance | directly threatens Hessian-based prune claims | does not jointly price the replacement destination |
| StructSplat `fit.py` | raw-absolute mean-gradient EMA, residual, support, responsibility | selector chooses rows; configured wave fixes the edit | existing differentiable 2D signals | no common action currency |
| StructSplat `safe_schedule.py` | responsibility/error/geometry proposals plus exact render | auctions transactional birth, split, prune/rebirth, merge/rebirth | already has fail-closed exact trials | proposal heuristics are not gradient action values |
| `realtime-gs` classic density | screen-gradient magnitude plus absolute world scale | small rows clone, large rows split; opacity/size prune | production topology seam and persistent lineage | scale/cadence can force the wrong operator |
| HIER-032 | explicit coverage debt plus contribution-aware donor search | donor-funded coverage transactions | proves destination coverage and donor damage must be coupled | every closure arm lost too much interior quality |

The local `realtime-gs` evidence is especially diagnostic. Only 2.8% of the control initializer's
original rows were split-eligible at the first density event. Their scales did grow by about 2.25x
per 1,000 steps, roughly 860 steps per doubling, but density control consumed its budget between
steps 20 and 500. The rows therefore cloned in place before crossing the absolute split threshold.
This is evidence of an initializer/controller clock mismatch, not evidence that gradients are
useless. It also shows why the current scale threshold cannot be treated as the inferred operator.

The open space is not “use gradients for densification”; that is established and crowded. The
plausibly useful remainder is to compare **count-balanced edits in one downstream loss currency**,
while explicitly measuring when source 2D evidence transfers to multi-view 3D evidence.

## 2. Functional problem signature

### 2.1 What the normalized 2D gradient says

For the normalized StructSplat renderer at pixel `p`,

\[
C_p = \frac{\sum_i w_{ip}c_i}{D_p}, \qquad
D_p = \sum_i w_{ip} + \varepsilon, \qquad
\rho_{ip} = \frac{w_{ip}}{D_p}.
\]

Ignoring the small derivative detail introduced by `epsilon`, a log-weight perturbation has

\[
\frac{\partial C_p}{\partial \log w_{ip}}
    = \rho_{ip}(c_i-C_p).
\]

For a geometric parameter `theta_i`,

\[
\frac{\partial C_p}{\partial \theta_i}
    = \rho_{ip}(c_i-C_p)
      \frac{\partial\log w_{ip}}{\partial\theta_i}.
\]

This exposes four limits of a raw row gradient:

- opposite pixel or view directions cancel under signed aggregation;
- absolute aggregation avoids cancellation but loses whether the evidence is coherent;
- when `c_i` is close to the current composite, geometry can have a weak gradient despite large
  residual error;
- a true support hole has no active basis function and therefore no row gradient at the missing
  destination.

At a fitted stationary point, useful rows also have gradients near zero. Prune and merge therefore
need removal/combination sensitivity, not the gradient of the unchanged model.

### 2.2 Why split is first-order invisible

Consider a function-preserving symmetric split that replaces one primitive by two half-mass
children at `theta + delta` and `theta - delta`. At `delta = 0`,

\[
\frac{d}{d\delta}
\left[\tfrac12 f(\theta+\delta)+\tfrac12 f(\theta-\delta)\right]_{\delta=0}=0.
\]

The first-order benefit cancels by construction. The useful signal begins at second order. This is
why a first-order positional gradient may find a stressed parent yet cannot certify that splitting
it is the correct topology change. SteepGS's splitting matrix is the relevant known solution family:
a negative minimum eigenvalue supplies both a split decision and a direction.

An exactly coincident, appearance-preserving clone is similarly non-identifiable: its two children
have the same function and initially receive the same gradients. It needs an explicit symmetry
break, an auxiliary gate, or a finite proposal followed by recovery.

### 2.3 The 2D-to-3D observability boundary

A 2D source field identifies image-plane location, scale, orientation, color, and local
approximation stress. It does **not** identify depth, occlusion order, or cross-view correspondence.
Splitting independent source fields can create extra components on unrelated camera rays and make
the 3D inverse problem less identifiable even when every source PSNR improves.

For 3D decisions, let `g_{ipv}` be the per-pixel positional subgradient for persistent Gaussian `i`
in training view `v`. Transport screen-space evidence into world coordinates with the camera
projection Jacobian `J_iv` before aggregation:

\[
G_i = \sum_{v,p} J_{iv}^{\mathsf T}g_{ipv}, \qquad
A_i = \sum_{v,p}\|J_{iv}^{\mathsf T}g_{ipv}\|,
\]

\[
\operatorname{GCR}_i = \frac{\|G_i\|}{A_i+\varepsilon}, \qquad
Q_i = \sum_{v,p}(J_{iv}^{\mathsf T}g_{ipv})(J_{iv}^{\mathsf T}g_{ipv})^{\mathsf T}.
\]

`G_i` measures net motion pressure, `A_i` measures collision-resistant stress, the coherence ratio
separates aligned from conflicting pressure, and `Q_i` retains directional structure. They should
be recorded beside responsibility mass/error, maximum responsibility, visibility count,
view-consistency, footprint, opacity, and parameter-family gradients. These are compact sufficient
statistics; retaining the full pixel-by-row Jacobian is unnecessary.

### 2.4 The correct action grammar at fixed count

At exact `N`, clone, split, and birth cannot be unary actions. The legal grammar is transactional:

- **teleport:** prune one low-cost donor and insert one destination row;
- **merge/rebirth:** merge two compatible rows and insert one destination row;
- **funded split:** prune one donor and replace one parent with two children;
- **balanced clone:** prune one donor and add one symmetry-broken child beside a retained parent;
- **pure prune:** only legal when the budget may shrink or when paired with another birth.

This grammar already exists in partial form in StructSplat's safe schedule. HIER-032's negative
result reinforces it: coverage placements were valuable, yet their donor funding damaged the
interior enough that no arm passed. Destination gain and source damage must be scored together.

## 3. Fixation anti-library

The following ideas are controls or known methods, not a new research contribution: increase the
gradient threshold; replace signed gradients with absolute gradients alone; choose split versus
clone from scale alone; densify high-residual pixels without contributor or visibility accounting;
prune low opacity alone; merge nearest rows in parameter space; teleport every inactive row to a
single-view residual maximum; or trust improved source-field PSNR as evidence of a better 3D
initializer. Also avoid training a predictor from the exposed HIER history before the discrete
labels are validated: FIT-020 already found that an early response-bend signal did not predict
held-out selection value.

## 4. Productive recombinations

### Candidate method P1 — Gradient-coherence operator router

**Central claim:** per-row absolute gradient mass and directional coherence can separate coherent
under-resolution, suited to a clone/move, from colliding subpixel demands, suited to a split.

**Novelty class:** N1.

**Known foundation:** AbsGS supplies cancellation-resistant absolute subgradients; GDAGS directly
uses gradient coherence to route cloning and splitting.

**Irreducible delta:** transport the statistics through persistent 2D-field lineage into the
downstream 3D coordinate system and evaluate them under StructSplat's exact-count transaction gate.

**New prediction:** high absolute mass with high cross-view coherence will favor a balanced clone
or local move, while high mass with low coherence will favor split candidates only when their
second-order test agrees.

**Cheapest killing test:** on frozen states, compare operator confusion and top-k regret against
the current gradient-plus-scale controller. **Null hypothesis:** coherence adds no predictive value
after scale, responsibility, and residual features are known.

**Prior-art threats:** GDAGS is a direct mechanism threat; AbsGS and conflict-aware densification
methods occupy adjacent space.

**Novelty confidence:** 5–15% for the router, 20–35% for its exact-count 2D-to-3D relationship;
primary-paper and official-code search only, cutoff 2026-08-12.

### Candidate method P2 — Splitting-matrix-certified funded split

**Central claim:** a funded split should be proposed only when the parent's splitting matrix has a
negative eigenvalue, with child displacement along its minimum-eigenvalue direction.

**Novelty class:** N1/N2.

**Known foundation:** Splitting Steepest Descent and SteepGS establish the second-order split
criterion; StructSplat supplies exact-count donor funding and containment certificates.

**Irreducible delta:** combine the known split certificate with a count-neutral donor and require
the whole transaction to pass containment, exact-render, and recovery gates.

**New prediction:** it will reject many high-absgrad parents selected by a magnitude rule and lower
split regret, especially near fitted saddle points.

**Cheapest killing test:** compute the matrix only for the top residual/absgrad parents in synthetic
two-lobe fixtures and frozen real states. **Null hypothesis:** the minimum eigenvalue does not rank
actual recovered split gains better than footprint scale.

**Prior-art threats:** SteepGS is direct; any novelty is limited to the balanced certified
transaction and transfer into normalized 2D fields.

**Novelty confidence:** 0–10% for split scoring, 15–30% for the task-specific relationship.

### Candidate method P3 — Balanced counterfactual transaction auction

**Central claim:** clone, split, merge, prune, and teleport candidates can be compared in one
currency by their predicted downstream loss change, provided the score covers the complete
count-balanced transaction rather than isolated donor or destination terms.

**Novelty class:** N2.

**Known foundation:** influence functions, Hessian-aware pruning, matching pursuit, transactional
StructSplat proposals, and exact trial-render acceptance.

**Irreducible delta:** define an action in rendered function space rather than parameter space. For
a legal edit `T_a`, let

\[
d_a = R(T_a(\theta))-R(\theta)
\]

be its **operator tangent** (a finite secant for discrete edits). With image-space loss gradient
`u = partial L / partial R`, its first-order predicted gain is

\[
s_a = -\langle u,d_a\rangle.
\]

A local quadratic model adds

\[
\widehat{\Delta L}_a = \langle u,d_a\rangle
  + \tfrac12\langle d_a,H_Rd_a\rangle.
\]

For squared error and a fixed rendered perturbation this term is exact; for the mixed 3D training
loss it is a generalized-Gauss-Newton or Hessian-vector approximation. Appearance/SH and mass are
locally refit before final scoring. The top few candidates then receive the existing exact render
and short-recovery gate.

**New prediction:** action scores will reduce top-k regret across operator families, and combined
donor-plus-destination scores will outperform separately ranking cheap donors and attractive births.

**Cheapest killing test:** the operator-identifiability oracle in section 10. **Null hypothesis:**
after equal compute, action-gradient or quadratic scores cannot beat existing residual,
responsibility, scale, opacity, and similarity heuristics.

**Prior-art threats:** rendering-aware pruning, neural growth, basis pursuit, and topology-search
literature may already instantiate close abstractions. No broad novelty claim is warranted.

**Novelty confidence:** 25–45% for the cross-operator, exact-count relationship; 0–10% for its
ingredients; cutoff and search scope as disclosed above.

### Candidate method P4 — Cross-view lineage gradient transport

**Central claim:** topology decisions made after 2D-field lift improve when per-view screen
gradients are mapped into a common world frame and aggregated by persistent 3D lineage, rather
than inherited independently from source fields.

**Novelty class:** N2/N3.

**Known foundation:** differentiable projection Jacobians, multiview gradient aggregation, and the
existing `realtime-gs` persistent-ID topology seam.

**Irreducible delta:** explicitly measure source-field versus downstream-world disagreement and
make cross-view agreement a gate, not merely another magnitude feature.

**New prediction:** source-only split/teleport proposals with poor cross-view agreement will have
negative or unstable downstream gains; coherent tangent-plane proposals will transfer better.

**Cheapest killing test:** rank the same bounded candidates using source 2D, single-view 3D, and
all-training-view 3D statistics, then compare actual held-view recovery. **Null hypothesis:** world-
frame aggregation provides no gain over ordinary view-space accumulation.

**Prior-art threats:** multiview density-control and view-consistency methods are close; a focused
search of 2D-to-3D Gaussian lineage control remains incomplete.

**Novelty confidence:** 20–40%, mostly in the relationship and diagnostic rather than the Jacobian.

### Candidate method P5 — Ghost insertion probes for holes and unseen destinations

**Central claim:** destination value can be differentiated by introducing auxiliary candidate
Gaussians with a zero-influence linear gate, even where no current row receives gradient.

**Novelty class:** N1/N2.

**Known foundation:** error backpropagation, matching pursuit, column generation, and GradMax-style
function-preserving growth.

**Irreducible delta:** use certified 2D sites or triangulated 3D sites as temporary renderer
columns, differentiate only an auxiliary linear gate, and discard the bank after ranking. Do not
differentiate a zero opacity logit, whose chain derivative can itself vanish.

**New prediction:** gate derivatives recover high-value birth/teleport destinations that row
gradients cannot see, while cross-view gate consistency rejects ray-only 2D artifacts.

**Cheapest killing test:** isolated missing-atom and multiview disocclusion fixtures. **Null
hypothesis:** ghost-gate rank is no better than residual maxima after exact action evaluation.

**Prior-art threats:** Revising Densification and neural-network growth methods cover much of the
mechanism; only its certified, balanced, persistent-lineage use may remain distinct.

**Novelty confidence:** 10–25%.

## 5. Exploratory candidates

### Exploratory candidate E1 — Multi-horizon operator oracle

**Central claim:** topology labels must be measured at immediate, 1-, 5-, and 20-step recovery
horizons because the best immediate edit may not remain best after optimizer recovery.

**Novelty class:** N1 measurement.

**Known foundation:** response spectroscopy and short-horizon architecture-growth evaluation.

**Irreducible delta:** enumerate all five operator families under matched count and recovery work.

**New prediction:** merge and teleport rankings will be less stable across horizons than local
clone rankings.

**Cheapest killing test:** deterministic synthetic fixtures. **Null hypothesis:** rankings are
horizon-invariant or no signal predicts any horizon.

**Prior-art threats:** hypergradient and influence-evaluation protocols.

**Novelty confidence:** 5–20%; value is evidentiary.

### Exploratory candidate E2 — Source/downstream disagreement atlas

**Central claim:** the sign and rank disagreement between source 2D and downstream 3D action gains
is a measurable predictor of unsafe field-derived topology edits.

**Novelty class:** N2 measurement.

**Known foundation:** domain-transfer diagnostics and multiview consistency analysis.

**Irreducible delta:** bind every 2D component to persistent 3D lineage and evaluate the same edit
under both objectives.

**New prediction:** boundary and high-frequency source improvements will show the largest
disagreement when correspondence/depth confidence is low.

**Cheapest killing test:** one sealed training/validation camera split on existing lifted scenes.
**Null hypothesis:** source/downstream disagreement is unrelated to held-view action value.

**Prior-art threats:** transferability and gradient-alignment diagnostics.

**Novelty confidence:** 20–35%.

### Exploratory candidate E3 — Camera-half action stability

**Central claim:** useful 3D topology actions remain high-ranked when the training cameras are split
into independent halves, whereas view-specific artifacts do not.

**Novelty class:** N1 measurement.

**Known foundation:** split-half reliability and cross-validation.

**Irreducible delta:** apply stability to discrete topology action values rather than model metrics.

**New prediction:** stable action rank will predict held-view recovery better than raw gradient
magnitude.

**Cheapest killing test:** compute top-k overlap and signed gain agreement across camera halves.
**Null hypothesis:** stability does not improve top-k precision.

**Prior-art threats:** ordinary resampling stability; no novelty claim for the statistical device.

**Novelty confidence:** 5–15%.

## 6. Transformational candidates

### Transformational candidate T1 — Operator-tangent field

**Central claim:** discrete density control can be formulated as optimization over a small set of
render-space operator tangents rather than classification of rows.

**Novelty class:** N3.

**Known foundation:** tangent methods, finite differences, matching pursuit, and trust-region
models.

**Irreducible delta:** the formal object is a typed edit plus its rendered secant, recovery horizon,
legality certificate, lineage transition, and optimizer-state transition.

**New prediction:** cross-operator ranking calibrated in render space will dominate independent
operator thresholds at equal proposal and trial budgets.

**Cheapest killing test:** P3's bounded oracle. **Null hypothesis:** a common render-space score is
not calibrated across operator families.

**Prior-art threats:** differentiable architecture search, edit gradients, and basis pursuit may
subsume the abstraction after deeper search.

**Novelty confidence:** 25–45%; provisional.

### Transformational candidate T2 — Capacity-transport graph

**Central claim:** exact-count topology control is better modeled as min-cost capacity transport
than as independent densification and pruning.

**Novelty class:** N3-T.

**Known foundation:** optimal transport, assignment, and StructSplat's funded edits.

**Irreducible delta:** donor rows are supply nodes, underfit sites are demand nodes, and feasible
split/merge/teleport transactions are typed hyperedges carrying removal cost, insertion gain,
certificate risk, and optimizer-state cost.

**New prediction:** joint assignment avoids the interior damage seen when attractive HIER-032
destinations are funded by separately chosen donors.

**Cheapest killing test:** solve a small exact bipartite/hypergraph instance from a frozen HIER
state and compare with greedy independent rankings. **Null hypothesis:** joint transport produces
no lower exact loss at the same count.

**Prior-art threats:** facility location, mixture reduction, and resource-allocation formulations.

**Novelty confidence:** 30–50% for the explicit Gaussian action graph, not the optimization tools.

### Transformational candidate T3 — Reversible Gaussian edit grammar

**Central claim:** topology search can preserve scientific auditability by representing every
operator as a reversible, typed rewrite with exact preconditions and receipts.

**Novelty class:** N3-T.

**Known foundation:** compiler rewrite systems, equality saturation, reversible-jump MCMC, and the
existing StructSplat transaction ledger.

**Irreducible delta:** equivalent edit sequences share a canonical lineage/state representation,
allowing the controller to compare, roll back, and cache render deltas without identity drift.

**New prediction:** canonicalized proposals reduce duplicate exact renders and make action outcomes
reproducible across controller schedules.

**Cheapest killing test:** canonicalize two equivalent prune/birth versus teleport sequences and
test field, lineage, optimizer-state, and render parity. **Null hypothesis:** canonicalization
cannot preserve all four or yields no useful cache reuse.

**Prior-art threats:** e-graphs, transactional optimizers, and reversible-jump samplers.

**Novelty confidence:** 20–40%; systems-level relationship only.

### Transformational candidate T4 — Gradient spectrum packet

**Central claim:** a small per-lineage packet `{G, A, Q, responsibility, leverage, visibility}` is
a sufficient online interface between a renderer and multiple topology controllers.

**Novelty class:** N2/N3.

**Known foundation:** gradient sketches, sufficient statistics, Fisher information, and GDAGS.

**Irreducible delta:** make the packet coordinate-aware, multiview, lineage-stable, and explicitly
operator-agnostic.

**New prediction:** the packet retains most exact-oracle top-k recall at far lower memory than
per-pixel gradient storage.

**Cheapest killing test:** feature ablation against full retained subgradients on small fixtures.
**Null hypothesis:** the packet loses too much action-ranking information.

**Prior-art threats:** rasterizer-side gradient-statistics methods may already expose similar
moments.

**Novelty confidence:** 15–35%.

## 7. Cross-domain transfers

| Source field | Preserved causal mechanism | StructSplat/3DGS transfer | Broken correspondences | Adoption barrier |
|---|---|---|---|---|
| Matching pursuit / column generation | add the basis column most correlated with the residual | ghost candidate gates rank births | normalized compositing and occlusion make columns state-dependent | candidate-bank render cost |
| Optimal experimental design / Fisher leverage | retain measurements or basis elements that support identifiable directions | prune low-leverage rows; protect sole-owner views/pixels | photometric loss is nonlinear and visibility changes | stable low-rank leverage approximation |
| Optimal transport / min-cost flow | jointly move conserved capacity from supply to demand | coupled donor/destination selection | split and merge are hyperedges, not simple unit flows | dynamic graph construction and certification |
| Neural architecture growth | use gradients/curvature to add function-preserving units | auxiliary gates and split curvature | Gaussians carry geometry, visibility, and explicit optimizer lineage | symmetry breaking without render shock |
| Influence functions / Hessian pruning | estimate loss after removal near a stationary point | prune and merge removal cost | topology edits can be large and nonlocal | HVP or approximation cost; REFINE prior art |
| Trust-region model-predictive control | propose, predict, measure, accept, and update the local model | top-k action trial and recovery gate | training objective drifts with camera minibatches | choosing a fair recovery horizon |
| Compiler superoptimization / e-graphs | enumerate typed rewrites, canonicalize equivalents, cache evaluations | reversible Gaussian edit grammar | floating-point fields are only approximately equivalent | state/optimizer canonicalization |
| Reaction networks / mass conservation | legal reactions conserve typed quantities | opacity/mass/count-preserving split/merge transactions | alpha compositing is ordered and not literal mass conservation | defining the conserved quantity correctly |
| Reserve markets / power dispatch | procure a scarce reserve jointly with delivery constraints | keep a capacity reserve for holes/disocclusions and auction it to demand | approximation error is not money and value is nonstationary | reserve size and opportunity cost |

The first six transfers are recognizable. Compiler rewrite systems, reaction networks, and reserve
markets are deliberately uncommon lenses. Their value is not analogy alone: each contributes a
testable mechanism—canonical edits, conservation preconditions, or explicit reserve opportunity
cost. None licenses a novelty claim without a deeper prior-art audit.

## 8. New-evidence discovery programs

### Evidence program D1 — Synthetic operator identifiability

Create fixtures with a known missing atom, a broad row covering two lobes, a duplicate pair, a dead
row, and a small coherently underfit row. Enumerate legal matched-count actions and measure actual
immediate plus short-recovery gains. The program succeeds only if the intended operator is actually
best under the rendered objective; otherwise the fixture assumption is wrong.

### Evidence program D2 — Frozen real-state counterfactual table

Save early, middle, and late states from both HIER and `realtime-gs`. For a bounded, preregistered
candidate set, record heuristic features, gradient packet, splitting eigenvalue, predicted action
gain, exact immediate gain, and 1/5/20-step gain. This produces operator confusion, regret, and
calibration evidence without committing a controller.

### Evidence program D3 — 2D-to-3D transfer matrix

For each persistent lineage, compare action ranks under source-field loss, fitted-view 3D loss,
independent camera halves, and sealed validation views. Stratify by depth/correspondence confidence,
boundary distance, visibility count, and occlusion. This is the decisive program for whether HIER
gradients are useful beyond source image compression.

### Evidence program D4 — Sufficient-statistic runtime audit

Instrument streamed `{G, A, Q}` and responsibility/visibility moments in the renderer. Report added
time, memory, determinism, and top-k recall relative to a small full-subgradient reference. The
packet is rejected if its overhead removes the real-time training advantage or if chunk/order
changes alter selected actions beyond a frozen tolerance.

## 9. Pareto analysis

Scores are 0–5; higher is better except cost, where lower is better.

| Candidate | Novelty | Falsifiability | Importance | Feasibility | Cost | Informative failure | Publication value |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 coherence router | 1 | 5 | 3 | 4 | 2 | 4 | 2 |
| P2 certified funded split | 2 | 5 | 4 | 3 | 3 | 5 | 3 |
| P3 counterfactual auction | 3 | 5 | 5 | 4 | 3 | 5 | 4 |
| P4 lineage transport | 3 | 4 | 5 | 3 | 4 | 5 | 4 |
| P5 ghost insertion | 2 | 5 | 4 | 4 | 2 | 4 | 3 |
| T2 capacity-transport graph | 4 | 4 | 5 | 2 | 5 | 5 | 4 |
| T3 reversible edit grammar | 3 | 4 | 3 | 2 | 4 | 4 | 3 |
| T4 gradient spectrum packet | 3 | 5 | 4 | 4 | 2 | 4 | 3 |

P3 is the best first target because it can reject the whole premise cheaply, uses existing exact
transaction machinery, and yields useful negative evidence. P1, P2, P4, and P5 should be measured
as signals inside its oracle, not shipped independently first. T2 becomes justified only if
separate donor and destination rankings are empirically suboptimal on the frozen table.

## 10. Recommended first experiment

### HIER-033 proposal — operator-identifiability oracle

Do not implement a live controller yet. Build a diagnostic that answers: **given a frozen state,
can gradient-derived signals identify which legal topology transaction will improve the downstream
3D objective?**

Freeze deterministic early/mid/late checkpoints from one StructSplat HIER sequence and at least one
`realtime-gs` field-derived 3D training sequence. Use only training cameras for signal construction;
keep validation views sealed until the candidate rankings and gates are frozen.

For every checkpoint, enumerate bounded candidate banks for:

- balanced clone: low-removal-cost donor plus a finite, symmetry-broken child near a retained row;
- funded split: donor plus two mass-preserving children along scale, GCR, and splitting-matrix
  directions;
- merge/rebirth: compatible pair plus residual/ghost destination;
- prune/birth teleport: low-removal-cost donor plus triangulated multiview destination;
- no-op control.

Locally refit color/SH and mass under identical work. Measure actual loss change immediately and
after 1, 5, and 20 optimizer steps. Compare these selectors under the same candidate bank:

1. current residual/responsibility/scale/opacity/similarity heuristics;
2. signed gradient norm;
3. AbsGS absolute gradient;
4. gradient coherence ratio;
5. split minimum eigenvalue;
6. first-order operator tangent;
7. quadratic/GGN operator tangent;
8. exact immediate trial.

Report per-operator AUROC, rank correlation, top-k recall, top-k regret, gain calibration, operator
confusion, count/certificate failures, runtime, and memory. Separately report source-2D rank versus
downstream-3D and held-view rank. Include camera-half stability and lineage receipts.

**Primary null hypothesis:** gradient-derived action scores do not reduce top-k regret relative to
the current heuristics at equal proposal and exact-trial budgets.

**Transfer null hypothesis:** source 2D action scores do not predict downstream 3D action gains once
correspondence confidence and visibility are controlled.

**Pass gate:** at least one preregistered gradient/action model improves top-k regret and top-k
recall on synthetic and real frozen states, retains its sign across the chosen recovery horizon,
and does not regress held-view action precision. The split-specific criterion must beat the scale
rule on split regret. Runtime must support a later sparse top-k trial schedule.

**Kill gate:** stop if exact trials themselves are unstable across recovery horizons, if no
gradient/action proxy beats existing heuristics, or if source-field rankings fail to transfer to
downstream 3D. A killed result still identifies whether HIER should remain only an initializer and
leave topology entirely to a downstream 3D controller.

No default changes, learned selector, count increase, held-view tuning, or outcome-dependent rescue
are permitted in this first experiment.

## 11. Audit limitations

This portfolio is mechanism research, not empirical validation. The local timing/eligibility
numbers are single-scene development evidence and do not establish a general failure of classic
ADC. The normalized 2D renderer and depth-ordered alpha compositor have different derivatives, so
the 2D equation above motivates measurements but is not a proof for 3D. SteepGS and GDAGS already
occupy much of gradient-informed split/clone control; LocoADC is a direct 2D regional densification
and merge threat; REFINE is a direct removal-sensitivity threat. The potentially distinct claim is
therefore narrow: a persistent-lineage, exact-count auction of complete topology transactions in a
shared downstream render-loss currency.

Patent, thesis, non-English, unpublished, and post-cutoff searches are incomplete. “Operator
tangent,” “capacity-transport graph,” and “gradient spectrum packet” are working names for formal
objects, not claims of first invention. Even a successful oracle would not establish end-to-end
quality, speed, or generalization; it would only justify implementing and benchmarking a live
controller in a later task.

## Practical operator interpretation

| Operator | Gradient-derived evidence | Required non-gradient or counterfactual gate |
|---|---|---|
| Clone | high absolute mass, high world-frame/view coherence, small footprint | finite symmetry break, donor cost, exact render/recovery |
| Split | high stress with conflicting directions or negative split curvature | minimum splitting-matrix eigenvalue, certified child geometry, donor cost |
| Merge | similar render/Jacobian columns and redundant view responsibility | local appearance refit plus exact pair-removal/merge cost |
| Prune | low leverage/responsibility and no sole-owner pixels/views | leave-one-out or Hessian removal cost; raw stationary gradient is insufficient |
| Teleport | low donor removal cost plus high ghost insertion derivative | triangulated/multiview-consistent destination and complete transaction trial |
| No-op | no candidate has stable positive predicted gain | always retained as the fail-closed baseline |

The short answer is therefore: **yes, gradients contain useful information about where and about
some aspects of which operator—but only after preserving per-pixel direction, mapping evidence
across views, adding curvature for split, and scoring finite balanced actions for merge/prune/
teleport.** The safest next move is to validate operator identifiability before altering HIER or
`realtime-gs` production density control.
