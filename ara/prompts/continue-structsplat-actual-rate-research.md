# Continuation prompt — StructSplat actual-rate research

**Execution status (2026-07-14): complete.** BENCH-007 Stage 1 failed its preregistered gate;
Stage 2, BENCH-008, and COMP-005 were not authorized. Preserve this prompt as the executed
contract, not as an instruction to rerun or tune the failed pilot. The terminal evidence is
`ara/evidence/bench007-stage1-killing-pilot-2026-07-14/run.md`.

Copy the prompt below into the next coding/research session. It is intentionally specific: “fix
everything” is not a measurable terminal condition, while this prompt defines the scientific
decision, authorized work, killing gates, and handoff artifacts.

---

You are the primary research engineer for the StructSplat repository. Continue autonomously toward
one outcome:

> Establish, with self-contained bitstreams and held-out evidence, whether tensor-driven
> blue-noise allocation contributes rate-distortion value beyond the strongest direct
> handcrafted and simple allocation controls; then either advance the mechanism or close/reframe
> the claim.

Do not optimize the old 168 KiB leaderboard. It is a completed high-rate local-policy diagnostic,
not a compression benchmark or SOTA result.

## Read before acting

Read these files completely, in this order:

1. `CLAUDE.md`;
2. `.claude/skills/core/SKILL.md`;
3. `.claude/skills/task-workflow/SKILL.md`;
4. `.claude/skills/benchmark/SKILL.md`;
5. `tasks/BENCH-007-actual-rate-structure-phase-diagram.md`;
6. `tasks/BENCH-008-common-native-causal-bridge.md`;
7. `tasks/COMP-005-decoder-synchronized-structural-geometry.md`;
8. `ara/evidence/storage-budget-168k-sota-audit-2026-07-13.md`;
9. `ara/evidence/research-portfolio-2026-07-13.md`;
10. `ara/logic/claims.md` and `tasks/INDEX.md`.

If you change a method, also load the method skill. Before declaring completion, load review and
docs-sync. Use the research-ideation skill again only if evidence forces a materially new research
direction; it does not replace task-workflow/benchmark implementation.

## Start-of-session safety

- Run `git status --short` and inspect every overlapping diff before editing.
- Preserve all user work and untracked evidence. Never reset, clean, or overwrite historical
  `ara/evidence/` artifacts.
- Treat a dirty tree as provenance that must be recorded, not as permission to absorb unrelated
  changes.
- Check current primary literature and official repositories at the execution date. Do not rely on
  paper summaries or the July 13 audit alone for claims that may have changed.
- Never label a local transplant with a paper method's name. Use
  `local_<mechanism>_control`; reserve native names for executed, provenance-pinned upstream code.

## Facts already established

- The completed fixed-storage run has 320/320 cells, but 168 KiB at its prepared resolutions is
  71.68–81.15 analytical bpp. SSPL1 streams are about 22 bpp and the target PNGs average about
  17.99 bpp. It cannot support a compression/SOTA claim.
- The broad novelty of structure-aware allocation, orientation, adaptive precision, progressive
  Gaussian coding, boundary gating, learned initialization, and clustered VQ is occupied or
  directly threatened by current work including Structure-Guided Allocation, Image-GS, SAD, SGI,
  GaussianImage++, P-GSVC, Contour-Aware 2DGS, WIPES, AIR, Instant-GI, and CGVQ.
- StructSplat's defensible current identity is a training-free, interpretable structural prior and
  causal testbed. Its publishable question is whether its specific tensor-metric/WSE mechanism
  survives actual-rate, direct-control, and native comparisons.
- `src/structsplat/codec.py` already writes self-contained SSPL1 streams.
  `benchmarks/rate_distortion.py` does not yet target byte caps, enforce equal candidate search,
  compute robust BD-rate, or produce the required held-out phase diagram.
- Kodak-24 is present locally but has been repeatedly used during method development; it is a
  replication set, not held out. DIV2K validation is the primary held-out confirmation set.

## Execute in this order

### 1. Freeze the BENCH-007 contract

Before viewing new outcomes:

- verify the DIV2K file IDs, license/source, hashes, original dimensions, and rate denominator;
- freeze Stage-0b rate-calibration IDs, Stage-1 DIV2K-training IDs, target hashes, target rates,
  resolution-normalized count formula/ladder, bit mixes, fit/QAT steps,
  checkpoint policy, seeds, primary endpoints, statistical plan, promotion gate, and abandonment
  rules in the run config and ARA evidence;
- add a dry-run command that prints the complete cell/search count and an estimated compute/storage
  envelope;
- clearly mark the four COCO fixtures as plumbing-only and Kodak as development-exposed.

If a protocol change is scientifically necessary after freezing, version the manifest and explain
the change before running; never rewrite the old contract.

### 2. Implement only the BENCH-007 substrate

Build a resumable target-rate benchmark around actual SSPL1 bytes:

- common-renderer arms exactly as specified in BENCH-007;
- independently fitted count candidates and identical codec/QAT search for every arm;
- complete-stream byte caps using original pixel dimensions;
- cold decode, in-memory/cold parity, central metrics, component bytes, timing, RSS/VRAM, hashes,
  dirty-tree provenance, explicit failure rows, and portable reports;
- nondominated monotone RD envelopes and BD-rate only on overlapping measured intervals, without
  extrapolation;
- image-cluster bootstrap after averaging correlated seeds, with the preregistered multiplicity
  correction.

Add focused unit tests first for byte-cap inclusion, bpp denominators, nonmonotone candidate
selection, missing targets, cold-decode corruption, envelope construction, no-overlap BD-rate, and
resume identity. Reuse existing benchmark helpers where their semantics match; do not grow the
already-large fair-density harness merely for convenience.

Do not implement BENCH-008 or COMP-005 while BENCH-007 Stage 1 is unresolved.

### 3. Run a bounded execution ladder

Run:

1. CPU/small-image unit and synthetic tests;
2. one-image GPU smoke with two arms and one rate;
3. Stage-0a plumbing and Stage-0b rate-calibration cells;
4. the frozen eight-image Stage-1 killing pilot.

Use the repository's resource wrapper and keep the user informed during long runs. Resume rather
than restart. If a cell fails, preserve the failure row and diagnose it; do not silently narrow the
matrix.

### 4. Make the preregistered decision

Compare tensor-WSE with the strongest direct nonlearned control, not only with the shipped default.
Apply the exact BENCH-007 gate.

- If Stage 1 fails, close or reframe the exact narrow compression claim. Do not tune that
  formulation on the eight pilot images to rescue it. A materially new mechanism requires its own
  task, null, and disjoint development screen. The negative result, mechanism maps, and
  implementation remain valid research output.
- If Stage 1 passes, freeze the expanded manifest and run DIV2K validation confirmation plus the
  Kodak replication. Do not call the pilot SOTA.
- If the result depends on renderer/objective interaction, enter BENCH-008 through task-workflow.
- If actual layout bytes are the binding loss and the structural mechanism survives, enter
  COMP-005 through task-workflow. Re-audit novelty before implementation.

### 5. Compare to current methods honestly

Maintain two separate tables:

- **common-mechanism controls:** shared StructSplat renderer/fitter/codec, useful for causality;
- **native-authentic methods:** official code, protocol, renderer, optimizer, rate definition, and
  environment, useful for external validity.

Prioritize native Structure-Guided Allocation, Image-GS, GaussianImage++, GaussianImage, SAD,
WIPES, AIR, and Instant-GI when code/checkpoints are available. Keep SGI and CGVQ in the
compression frontier even if their code is unavailable. A parameter-count or analytical BPP row
must never occupy an actual-codec BPP column. Do not create a synthetic cross-paper leaderboard
when datasets, rate definitions, or objectives differ.

Include ordinary codec controls appropriate to the claim (at minimum lossless PNG as a sanity
check and published/available learned or conventional codec curves for context). StructSplat is
not “image compression SOTA” unless it is tested against the broader codec frontier under matched
metrics and rate.

## Scientific rules

- State a null, mechanism prediction, direct prior-art threat, cheapest killing test, promotion
  rule, and abandon rule before every new experiment.
- Count all transmitted information. Header, ranges, codebooks, masks, base layers, model weights
  specific to an image, and side information are not free.
- Never substitute Gaussian count, float payload, checkpoint size, or a paper formula for actual
  self-contained bytes.
- Do not infer causality from native end-to-end differences. Use explicit interventions or label
  the result descriptive.
- Do not promote from aggregate mean alone. Preserve per-image failures, confidence intervals,
  search multiplicity, and compute regressions.
- Negative evidence closes the exact tested formulation; it does not refute materially different
  mechanisms.
- Historical evidence is immutable. New runs get new dated evidence directories.

## Definition of complete

This continuation is complete only when:

- BENCH-007's target-rate harness and focused tests pass;
- the frozen Stage-1 pilot is fully run or a concrete, evidenced external blocker prevents it;
- the preregistered gate is applied without post-hoc rescue;
- raw rows, streams/hashes, configs, resource telemetry, failure visibility, statistics, and a
  portable report are saved;
- task status, index, claims, README/benchmark docs, and ARA evidence agree;
- native and local-control labels/rate definitions are audited;
- unrelated dirty work is preserved; and
- the final handoff states what was established, what was refuted, what remains unknown, and the
  single next authorized task.

Do not claim “all problems fixed.” Name the verified scope and leave explicit tasks for everything
outside it.

---
