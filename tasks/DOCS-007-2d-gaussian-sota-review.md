# DOCS-007 — 2D Gaussian image-field state-of-the-art review

## Context
The 2D Gaussian image-representation literature has expanded rapidly through July 2026, while the
repository's deepest standalone audit has a 2026-07-13 cutoff. A current review is needed that
separates reconstruction, encoding convergence, rendering performance, actual rate, and file-size
reduction rather than collapsing incomparable author-reported results into one leaderboard.

## Goal
Produce a primary-source literature review, current through 2026-08-04, that identifies the
Pareto-relevant methods and mechanisms for mapping images to 2D Gaussian fields and gives a
claim-safe research and engineering recommendation for StructSplat.

## Non-goals
- Do not run or claim a new native benchmark.
- Do not promote a StructSplat method, default, novelty claim, or compression result.
- Do not treat analytical parameter bits, checkpoint size, or original-file reduction as actual
  rate-distortion evidence.

## Acceptance criteria
- [x] Search scope, cutoff, inclusion criteria, query families, and evidence limitations are explicit.
- [x] The review covers representation/renderer, allocation, optimization/amortization, systems,
      quantization/entropy coding, progressive/random-access, and richer-primitive controls.
- [x] A source-backed comparison records quality, convergence/encode time, render/decode
      performance, storage semantics, and original-file compression semantics without a false common
      leaderboard.
- [x] The synthesis identifies Pareto leaders by operating regime and maps each transferable
      mechanism to the existing StructSplat task graph.
- [x] The evidence index, task state, and generated session brief are synchronized.
- [x] Structural checks and `./scripts/verify.sh` pass.

## Interfaces touched
`ara/evidence/2d-gaussian-image-fields-sota-review-2026-08-04.md`, `ara/evidence/README.md`,
`tasks/INDEX.md`, `tasks/SESSION-BRIEF.md`.

## Depends on
BENCH-005, BENCH-007, COMP-013, BENCH-025.

## Agent workflow
- Driver: codex-root
- Reviewer: pending-distinct
- Turn: reviewer
- Reviewed revision: report blob `40b771e29c58d20015b4b736300c6a6ba2695eae`

### Handoff log

### Handoff

#### Objective
Deliver a deep, current, primary-source review of image-to-2D-Gaussian methods that separates
quality, convergence, rendering/query performance, complete storage, and reduction against an exact
original file, then turn the literature into claim-safe controls for StructSplat.

#### Changes
Added the 8,124-word evidence artifact at
`ara/evidence/2d-gaussian-image-fields-sota-review-2026-08-04.md`. It defines strict-Gaussian and
adjacent-method boundaries, four evidence levels, complete-byte and original-file-ratio equations,
a 22-method taxonomy, source-qualified quantitative tables, regime-specific Pareto frontiers, a
cold-package evaluation protocol, mechanism synthesis, deployment recommendations, occupied claim
space, and an exact mapping to the Field V2 task graph. Added the artifact to the evidence index and
synchronized the generated session brief. No code, renderer, method, codec, default, ADR, or claim
ledger row changed.

#### Evidence
The literature search and forward/backward source sweep was current through 2026-08-04. Exact table
values were rechecked against locally extracted primary PDFs for GaussianImage, Image-GS, LIG,
EigenGS, Instant-GI, WIPES, GaussianImage++, SmartSplat, Structure-Guided Allocation,
Contour-Aware 2DGS, Fast-2DGS, EllipssianNet, SGI, GaussianVision, P-GSVC, SAD, AIR, CGVQ,
LocoADC, and GTC. PA-G2DS is deliberately limited to abstract/metadata because its full primary
table was unavailable. The report's SHA-256 is
`fa43546990b77a91fa24c54a5901dd8b1c3248ac092788699a42d288ad079a51`. `./scripts/verify.sh`
passed with 1,560 tests, 4 skips, 514 deselections, and all docs, ARA, task-policy, script-layout,
and agent-workflow checks clean.

#### Assumptions
Author-reported values are comparable only inside their source protocols. “Strict Gaussian” means
the decoded image is generated from 2D Gaussian kernels; SAD, WIPES, GTC, and PA-G2DS are labelled
adjacent controls. A deployed shared predictor may be excluded from per-image bpp only when its
version and installation cost are reported separately with an explicit amortization analysis.

#### Uncertainties
No reviewed source supplies both a complete cold-decodable Gaussian package and the exact supplied
source-file bytes needed to establish the requested original-file compression-ratio frontier. SGI
is the clearest complete high-resolution package found, but its paper tables still do not provide
that denominator. PA-G2DS quantitative claims remain unpromoted. Rapid 2026 preprint evolution and
cross-paper protocol differences preclude a universal leaderboard.

#### Review focus
Independently verify the Level-A/Level-B classification, especially GaussianImage bits-back and
Structure-Guided/CGVQ inherited coding boundaries; verify that SGI counts all per-image generative
state; inspect SmartSplat's seven-byte/raw-RGB assumption; challenge the Pareto labels for hidden
training, model, cache, or mask costs; and repeat the 2026-07-30 through 2026-08-04 date sweep for
missed work.

#### Protected actions not taken
No native external benchmark, sealed outcome, result promotion, new method, default change,
scientific claim acceptance, commit, push, or sibling-repository write was performed. BENCH-005,
BENCH-007, COMP-013, and BENCH-025 remain separate authorities. Existing unrelated IDE changes were
left untouched.

#### Recommended next action
A reviewer distinct from `codex-root` should audit the frozen report blob and append a structured
review. Independently of that review, COMP-013 should implement the direct complete-byte baseline;
only BENCH-025 may authorize an SGI-like seed grammar, and a future original-file benchmark should
use the report's cold-package protocol.

## Notes
Author-reported values remain labelled as such. Repository-native evidence and local analogue
experiments are separate evidence tiers. Independent scientific review is required before the
review's synthesis is described as accepted.
