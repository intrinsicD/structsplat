# Deep-research prompt — StructSplat publication readiness and visual evidence

Use this prompt with an agent that can inspect the live StructSplat repository, browse primary
literature, run code, and edit the repository. The literature cutoff is the execution date; record
the exact date in the output.

---

You are the adversarial research lead and research-software engineer for StructSplat. Work from the
live repository, not from a generic description. Your objective is to determine the narrowest
publishable scientific story that survives current prior art, identify every missing proof and
visual needed to communicate or falsify it, and implement deterministic figure-generation code for
the highest-value missing method/diagnostic visuals.

The current candidate claim is intentionally narrow:

> Under a common normalized 2D-Gaussian renderer, optimizer, and self-contained codec, does a
> training-free structure-tensor metric plus weighted sample elimination improve actual-rate
> distortion beyond direct SLIC/Sobel, gradient, uniform-WSE, and random allocation controls in
> the sparse regime?

Do not broaden this into “structure-aware Gaussians are novel” or “StructSplat is SOTA.” Treat the
pre-registered BENCH-007 actual-rate experiment as planned work, not completed evidence.

## 1. Read and map before proposing

Read completely:

1. `CLAUDE.md` and the applicable repository skills;
2. `README.md`, `docs/{architecture,theory,comparison}.md`;
3. `src/structsplat/{structure_tensor,density,sampling,init,gaussians,render,codec}.py`;
4. `tasks/INDEX.md` and `tasks/BENCH-007-actual-rate-structure-phase-diagram.md`;
5. `ara/logic/{problem,claims}.md`;
6. `ara/evidence/research-portfolio-2026-07-13.md` and
   `ara/evidence/storage-budget-168k-sota-audit-2026-07-13.md`;
7. existing tests, benchmark reports, figures, and image assets.

Construct a live frontier map covering primitive objects, assumptions, objective, renderer
semantics, initialization, optimization, codec rate definition, evidence, failure modes, and
unresolved anomalies. State any repository/profile conflict explicitly.

Rewrite the method without domain nouns: inputs, latent state, local/global coupling, continuous
and discrete variables, invariances, partial observability, finite resources, and failure modes.
List the fixation anti-library: obvious or already-tested ideas that must not be sold as novelty.

## 2. Perform real, adversarial source research

Search primary papers, proceedings pages, author/project pages, official repositories, theses, and
patents where accessible. Search the recipient field, donor fields, and bridge fields. At minimum,
audit these families and their newest successors:

- 2D Gaussian image representations and codecs: GaussianImage, Image-GS, GaussianImage++,
  Structure-Guided Allocation, Soft Anisotropic Diagrams, SGI, AIR, P-GSVC, CGVQ,
  Contour-Aware 2DGS, WIPES, Instant-GI, and learned image-guided samplers;
- structure tensors and anisotropic sampling: tensor scale-space, anisotropic blue noise,
  weighted sample elimination, adaptive approximation, and error-controlled meshing;
- outside-class compression controls: current conventional and learned image codecs relevant to
  actual self-contained rate claims;
- donor mechanisms: minimum-description length, optimal experimental design, statistical-physics
  phase diagrams, responsibility/mixture diagnostics, and predictive/scalable coding.

Search exact terms, synonyms, equations, functional descriptions, older terminology, and official
code. For every serious claim, identify the nearest work by facet: problem, representation,
mechanism, prediction, and evidence. Try to reconstruct StructSplat from one work and then from a
combination of works. Report the irreducible remainder.

Never fabricate a citation, result, availability claim, or novelty conclusion. Prefer
“apparently unexplored under this search and cutoff” over absolute novelty. Separate donor-method,
correspondence, adaptation, and prediction novelty. Give a confidence range and the strongest
prior-art threat. Cross-paper metric numbers are context, not a synthetic leaderboard.

## 3. Run independent idea/evidence lanes

Preserve distinct lanes before ranking:

- productive recombination;
- assumption surgery;
- primitive/grammar change;
- new-evidence programs;
- cross-domain mechanism transfer from at least four donor fields, including at least two rarely
  linked to 2D Gaussian image representations and one measurement/protocol transfer.

Apply the A+B, subtraction, grammar, prediction, necessity, and compression tests. For transfers,
also apply terminology-removal, structural correspondence, causal preservation, counter-analogy,
native-baseline, and historical-obviousness tests. Downgrade generic combinations.

Retain the current actual-rate phase diagram unless new evidence kills it. For every surviving
high-novelty candidate, specify the cheapest killing test, null, positive signature, strongest
conventional explanation, compute/data, decisive plots, confounders, and abandonment rule. Do not
implement a new scientific method without a registered task and explicit evidence gate.

## 4. Audit publication readiness claim by claim

Build a claim-evidence matrix with four labels: supported now, implemented but unvalidated,
planned/preregistered, and absent. A paper is not publication-ready merely because code exists.
Audit at least:

- novelty and relationship to closest work;
- actual-rate held-out evidence and direct controls;
- native external validity;
- rate accounting and cold-decode parity;
- statistical design, uncertainty, and negative results;
- scaling, runtime, memory, reproducibility, release/licensing, and artifact usability;
- manuscript completeness, equations, algorithm specification, limitations, and ethics/data
  statements where applicable.

For each gap, state whether it blocks submission, blocks a strong claim, or is a polish item. Give
the exact experiment, artifact, or text needed to close it. Do not claim a venue guarantee; if no
venue is specified, assume a top graphics/vision conference contribution bar and state that
assumption.

## 5. Perform a visual-evidence audit

Inspect figures in the nearest primary papers and identify the communicative role of each figure,
without copying their artistic expression. Produce a paper-figure plan with, at minimum:

1. method overview and representation anatomy;
2. structure-tensor anatomy on a real image: energy, coherence, flat/edge/corner labels, tangent
   orientation glyphs, and the exact density used downstream;
3. tensor-metric sampling anatomy: candidate/site distribution, local exclusion metric, selected
   WSE sites, and initialized Gaussian ellipses;
4. normalized-renderer anatomy: denominator/coverage, effective contributor count,
   responsibility entropy, dominant owner, and reconstruction;
5. causal allocation comparison at identical count/rate with zooms and identical visualization
   scaling;
6. actual-rate RD phase diagram with raw per-image points, intervals, and stream-component bytes;
7. edge/texture mechanism figure including signed cross-edge bleed;
8. optimization/convergence and resource scaling;
9. representative successes, ordinary cases, and failures selected by a predeclared rule;
10. supplemental ablations, sensitivity, and cold-decode/provenance checks.

For every proposed panel specify: claim served, input data, exact computation, visual encoding,
comparability constraints, caption takeaway, and status (implemented/data missing/result missing).
Distinguish explanatory method diagrams from empirical result figures.

## 6. Implement only truthful, high-value figure infrastructure

Implement deterministic, tested code for visuals that can be computed from the existing method
without inventing missing experimental results. At minimum it must:

- load a real image in the repository or a user-supplied image;
- use the production `structure_tensor`, `density`, `init`, and normalized renderer paths;
- visualize robustly scaled tensor energy and density without implying absolute units;
- display coherence, labels, tangent glyphs, WSE sites, and RS Gaussian ellipses with documented
  `(x,y)` and angle conventions;
- compute normalized-renderer diagnostics from the exact clipped-support weight equation:
  denominator/coverage, effective contributor count, responsibility entropy, dominant owner, and
  reconstruction;
- save lossless individual panels plus one labeled montage, a machine-readable manifest, resolved
  configuration, source-image hash, and deterministic output hashes;
- expose CLI help and accept seed, count, strategy, tensor settings, crop/max-side, glyph spacing,
  ellipse limit, and output directory;
- avoid requiring CUDA; preserve the NumPy/torch split; never alter the method or fit silently;
- mark initialization-only output as initialization, not optimized quality evidence.

Add focused tests for determinism, dimensions/ranges, label colors, tensor orientation convention,
diagnostic identities (responsibilities sum to one where covered; effective count is bounded),
manifest provenance, and a CLI smoke test. Generate one example evidence directory from a pinned
repository image. If dependencies or runtime prevent a requested panel, record the gap rather than
fabricating it.

## 7. Deliverables

Save:

- this prompt under `ara/prompts/`;
- a dated, source-linked research/publication audit under `ara/evidence/`;
- a bounded task record for the visualization implementation;
- figure code in a reusable package module plus a thin script/CLI entry point;
- tests and a reproducible example evidence directory;
- documentation explaining how to regenerate the figures and what they do **not** establish.

Conclude with:

- the narrowest defensible paper claim today;
- the single decisive experiment still missing;
- the implemented figure set;
- the remaining figure/data queue in priority order;
- a Pareto shortlist and the cheapest next killing experiment;
- audit limitations, inaccessible sources, and literature cutoff.

Do not call the work publication-ready unless every blocking row in the claim-evidence matrix is
actually closed.

---
