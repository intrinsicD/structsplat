# CORE-018 geometry-recovery portfolio

Date: 2026-08-06

Literature cutoff: 2026-08-04 local audits plus the live StructSplat/realtime-gs trees

Scope: planning evidence only; no novelty, quality, speed, or compression claim

## 1. Frontier map

| Family | Primitive and mechanism | What current evidence says | Remaining failure |
|---|---|---|---|
| CompactCarve interior consensus | Independent source rays; pointwise color variance over compact fields | Provides a usable initialization but consumes large sparse-index work | Broad depth peaks become volumetric centers, halos, and floaters |
| Alpha-shell placement | First depth attaining maximal multiview mask support | CORE-017 improves PSNR and alpha localization | Visual hull cannot resolve true surface; trailing doubles remain |
| Component Splat-SfM | Discrete fitted-splat matches, tracks, DLT, covariance LSQ | Exact on consistent synthetic fields | Per-view decompositions are not repeatable detections; real floaters remain |
| Raw patch/epipolar graph | Reciprocal distinctive local patches plus exact calibration | Broadens graph coverage | Strict surface-contribution precision failed before optimization |
| Monocular depth lift | Shared learned inverse-depth prior aligned to scene bounds | Fast and available in realtime-gs | Per-view affine depth ambiguity and cross-view inconsistency |
| Feed-forward multiview geometry | Learned depth/point maps or dense correspondence | Strongest external speed/quality family | Large prior, domain dependence, and no current packet-native receipt path |
| Ordinary 3DGS optimization | Photometric descent plus density control | Can recover good endpoints from weak starts | Wrong geometry costs convergence and may leave floaters/blur |

The unexplained residual is directional: alpha-shell centers are consistently nearer the visible
surface, yet a depth-offset copy survives along view rays.  That is evidence against another mask,
opacity, covariance, or optimizer-weight treatment and for a source-excluded depth observable.

## 2. Functional problem signature

Inputs are compressed continuous observations and calibrated projective maps.  Hidden state is a
set of visible 3D surfaces plus view-conditioned appearance and occlusion.  The system must allocate
a finite number of renderable kernels, infer a continuous depth for each accepted source event,
reject explanations visible in too few compatible observations, preserve discontinuities, and
serialize the result below the original-file budget.  Depth is continuous; proposal acceptance and
topology are discrete.  Geometry is globally coupled through projection but evidence and occlusion
are local.  Repeated texture, specularity, low texture, boundaries, and missing views limit
identifiability.

## 3. Fixation anti-library

The rejected defaults are: add another loss weight; optimize longer; copy alpha to opacity; use a
larger covariance; merge nearby splats; run SIFT/Splat-SfM unchanged; infer one monocular depth map
per view and concatenate; use every view in an ordinary mean; or call a larger neural backbone a
method.  Each can be a control, but none directly represents occlusion-aware depth uncertainty.

## 4. Productive recombinations

### R1 — Occlusion-aware ray posterior (selected)

- **Claim:** A source-excluded posterior over bounded ray depth, using robust best-view evidence and
  reciprocal agreement, removes isolated depth modes that alpha support and all-view color variance
  retain.
- **Class:** N1 systems recombination; no novelty claim.
- **Irreducible delta:** uncertainty and occlusion are represented before a Gaussian mean exists.
- **Prediction:** narrower posterior/more reciprocal support predicts cleaner step-zero geometry and
  earlier reporting-view PSNR without a larger field.
- **Kill:** equal-packet/equal-count three-arm disjoint scene; abandon on any retained double trail.
- **Threat:** plane-sweep MVS and robust stereo already own the components; value is only measured
  packet-to-realtime-gs behavior.

### R2 — Monocular proposal, multiview correction

- **Claim:** A shared monocular inverse-depth prior can reduce search work if calibrated views only
  correct a bounded interval and reject inconsistent pixels.
- **Class:** N1.
- **Delta:** feed-forward depth is a proposal distribution, never accepted geometry.
- **Prediction:** at equal final quality, depth samples fall by at least 4x; without correction,
  per-view scale seams remain.
- **Kill:** compare full-range and prior-window posterior on synthetic scale perturbations.

### R3 — Track anchors plus edge-aware depth completion

- **Claim:** High-confidence multiview tracks can anchor a piecewise-affine inverse-depth field that
  fills textureless support without forcing false correspondences.
- **Class:** N1/N2.
- **Delta:** unmatched observations are coverage debt, not geometry.
- **Prediction:** completion helps smooth cloth but fails at unobserved depth discontinuities unless
  the graph uses appearance boundaries.
- **Kill:** two-plane occlusion fixture with texture removed from one plane.

### R4 — Geometry-first staged optimizer

- **Claim:** Freezing accepted means for a short appearance/opacity solve before density growth
  preserves useful geometry and improves time-to-quality.
- **Class:** N1.
- **Delta:** initialization confidence controls parameter activation, not only learning rate.
- **Prediction:** early fixed-topology gain survives the density phase; immediate full optimization
  erases it.
- **Kill:** matched attempted steps and wall-clock with/without the freeze.

## 5. Exploratory candidates

### E1 — Multi-modal layered ray state

Retain two well-separated posterior modes at occlusion boundaries and let differentiable visibility
select between them.  This changes one-ray/one-mean placement but risks doubling capacity.  Kill it
if a two-mode arm cannot beat one mode at equal complete bytes on a thin foreground fixture.

### E2 — Uncertainty-priced topology

Allocate rows according to expected held-out distortion reduction per encoded byte, using posterior
entropy, image frequency, and codec cost.  It is only useful after a real final-model codec exists.
Kill it if uncertainty ranking is uncorrelated with post-fit error under fixed count.

### E3 — Learned packet-native depth head

Distil the expensive posterior into a small elastic head over decoded packet features.  The shared
model is amortized but must report training/model/break-even cost.  Kill it if a direct posterior is
already below the desired latency or if the head loses disjoint-scene consistency.

### E4 — Temporal geometry reuse

For fixed cameras, transmit/optimize a reference geometry field and per-frame appearance plus sparse
topology deltas.  This likely dominates per-frame reconstruction on video, but it changes the unit
of compression.  Kill it on two distant karate frames if shared geometry plus deltas is not smaller
than two independent final models at equal reporting quality.

## 6. Transformational candidates

### T1 — Visibility explanations instead of independent Gaussians

- **Class:** N3 candidate.
- **Changed grammar:** the latent unit is a surface explanation with a posterior over visibility;
  render Gaussians are compiled outputs, not optimized primitives.
- **Prediction:** one explanation may emit different LOD kernels without changing geometry or
  visibility.
- **Necessity:** independent splats conflate existence, coverage, opacity, and confidence.
- **Kill:** compile one explanation into two counts and test geometry/alpha consistency.

### T2 — Query-conditioned scene code

- **Class:** N3 candidate.
- **Changed grammar:** store a compact spatial program that emits Gaussians for a requested camera
  frustum and byte budget rather than one global flat array.
- **Prediction:** visible-byte cost scales with requested view coverage, not total scene extent.
- **Kill:** random-access render latency and complete bytes versus one flat 3DGS stream.

### T3 — Conservation of geometric support mass

- **Class:** N3 candidate.
- **Changed grammar:** split/merge/densify conserve a non-rendering support measure distinct from
  opacity; topology operates on that measure while appearance remains free.
- **Prediction:** topology becomes less sensitive to opacity resets and view sampling.
- **Kill:** identical render initialization with/without conserved mass under repeated split/prune.

## 7. Cross-domain transfers

### X1 — Robust sensor fusion with missed-detection state

| Role | Donor | Recipient |
|---|---|---|
| State | Object state distribution | Ray-depth distribution |
| Observation | Noisy sensor return | Projected packet feature |
| Operator | Likelihood update | Calibrated feature comparison |
| Invariant | Probability including missed detection | Posterior including dustbin |
| Failure | Clutter/occlusion | Repeated texture/hidden surface |

The preserved mechanism is that a missing or adversarial observation must not drag an arithmetic
mean.  Broken correspondences are view-dependent appearance, correlated cameras, and no calibrated
noise model; all are correctable only empirically.  Prediction: best-K plus dustbin beats all-view
mean specifically when one view is occluded.  Native competitor: ordinary CompactCarve consensus.

### X2 — Error-correcting consensus (rare donor)

Depth hypotheses are codewords; independent views are parity checks; reciprocal reprojection is a
syndrome.  The transfer preserves rejection through structured redundancy, but views are not
independent, a wrong repeated texture can satisfy several checks, and there is no discrete channel.
Required invention is a calibrated continuous syndrome.  Prediction: reciprocal support separates
coherent surfaces from floaters even when raw likelihood margins overlap.

### X3 — Sheaf-like local-to-global consistency (rare donor)

Each source ray owns a local depth section; camera reprojection maps overlapping sections; accepted
geometry requires agreement on overlaps.  The mechanism is global consistency from local
restriction maps.  Occlusion means some overlaps intentionally have no section, projection is
many-to-one, and discretized candidate graphs are incomplete.  Prediction: inconsistency cycles
localize the views/rays responsible for double surfaces better than terminal RGB error.

### X4 — Sequential probability ratio testing

Order target views by expected information and stop scoring a ray once one depth is decisively
separated or every mode falls into the dustbin.  This imports a measurement protocol, not a new
representation.  View evidence is correlated and likelihoods are approximate, so thresholds need
calibration.  Prediction: easy textured rays use fewer view-depth queries without changing their
winner; hard rays consume the full budget and remain explicitly uncertain.

## 8. New-evidence programs

### P1 — Posterior pathology atlas

- **Search:** vary texture repetition, specularity, boundary angle, occluded-view count, baseline,
  codec quality, and depth sampling on synthetic layered scenes.
- **Observable:** winner error, entropy, mode separation, reciprocal syndrome, and rendered trail
  energy—not only PSNR.
- **Conventional expectation:** confidence margin should rank correctness.
- **Surprise:** a stable regime where entropy is low but reciprocal syndrome is high would show
  confident cross-view inconsistency and justify a new uncertainty model.
- **Controls:** exact cameras/depth, uncompressed-image oracle, shuffled views, repeated seeds,
  reference projection, finite-difference coordinate tests.

### P2 — Artifact-to-geometry attribution

- **Search:** hold final appearance fixed while swapping only interior, shell, posterior, and oracle
  means/covariances at equal count.
- **Observable:** directional error autocorrelation aligned with source rays, alpha leakage, and
  depth-support width before/after optimization.
- **Conventional expectation:** better geometry should reduce ray-aligned trail energy at step zero.
- **Surprise:** unchanged trail under oracle means would redirect effort to opacity/appearance or
  renderer semantics rather than depth.
- **Controls:** exact same SH/opacity/count, source-view exclusion, native pixel inspection, and
  renderer parity checks.

## 9. Pareto frontier

Scores are planning judgments from 0 to 5, not evidence.

| Candidate | Novelty | Falsifiable | Importance | Feasible | First-test cost | Informative failure | Publication potential |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 ray posterior | 1 | 5 | 5 | 4 | 3 | 5 | 3 |
| R2 monocular proposal | 1 | 5 | 4 | 4 | 3 | 4 | 2 |
| R3 track completion | 2 | 4 | 4 | 3 | 3 | 4 | 3 |
| E1 layered modes | 3 | 5 | 5 | 3 | 3 | 5 | 4 |
| E3 distilled head | 2 | 4 | 5 | 2 | 1 | 3 | 4 |
| T1 explanations | 4 | 4 | 5 | 2 | 1 | 5 | 5 |
| T2 query code | 4 | 4 | 5 | 1 | 1 | 4 | 5 |
| X4 sequential test | 2 | 5 | 4 | 4 | 4 | 5 | 3 |

## 10. Recommended first experiment

Implement R1 as a bounded reference path.  First kill the scoring rule on an exact two-surface
synthetic fixture with repeated texture and one occluded view.  If it survives, run one immutable
three-arm diagnostic on the previously unused unmasked `karate/frame_00060`: interior consensus,
posterior without reciprocal agreement, and full posterior, all from identical packets and through
the same staged realtime-gs optimizer.  The null is that feature evidence and reciprocity do not
produce cleaner geometry or faster quality than ordinary consensus.  Any persistent reporting-view
double silhouette abandons R1 on that scene without threshold rescue.

## 11. Audit limitations

This portfolio reuses the repository's 2026-08-04 literature audits and current realtime-gs
negative results; it does not perform a new independent prior-art search.  DINO features may be too
coarse for thin structures, the karate scene has no masks or ground-truth depth, and one scene
cannot establish generality.  Complete packet/model byte accounting does not include a future
production entropy codec or the amortized model-download/training break-even.  The selected method
is valuable only if measured behavior survives these limitations.
