# FF-002 — Field V2 Predict–Optimize–Distil

## Context

FF-001 established a bounded warm-start signal but its global-pooled, fixed-row predictor discards
spatial layout and uses row-wise MSE for an unordered Gaussian set. Its evidence also predates the
Field V2 semantic decision, downstream objective, production recipe, and complete-byte codec. The
next learned method must predict the selected semantic object, keep rendered appearance
authoritative, and learn from short optimizer corrections without backpropagating through an
unbounded fitting trajectory.

## Goal

A spatial, permutation-invariant predictor for one selected Field V2 operating point, developed as
four matched comparators and able to initialize the frozen iterative recipe for `0/50/200/500`
refinement steps.

## Candidate comparators

- **A:** frozen FF-001 predictor/recipe through an explicitly labelled compatibility adapter.
- **B:** local grid/candidate predictor plus permutation-invariant field supervision.
- **C:** B plus authoritative Field V2 render loss.
- **D:** C plus Predict–Optimize–Distil targets produced by short runs of the BENCH-021 recipe.

The first architecture is a compact U-Net/FPN producing a spatial candidate map with occupancy and
field attributes followed by deterministic top-N selection. A small query decoder is a recorded
fallback only if a preregistered capacity diagnostic shows grid underfit; it is not another
outcome-visible arm.

## Loss and teacher contract

Rendered RGB under the BENCH-020 equation is authoritative. Set matching may supervise geometry
and `rgb_coeff`, but multiple pixel-equivalent decompositions must not be penalized as incorrect.
`structural_mass`, alpha, density, or downstream-surrogate terms are present only if BENCH-019/020
defined and validated them. The teacher is the exact BENCH-021 recipe encoded by COMP-013; teacher,
short-refined student, and hand initializer roles stay separate and are never averaged into one
ambiguous row target.

## Non-goals

- Backpropagating through optimizer trajectories, elastic budgets (FF-003), or changing defaults.
- Training a neural decoder required at realtime-gs query time.
- Row-wise teacher matching as the authoritative loss or normalized/additive semantic relabelling.
- Formal product-speed claims; BENCH-023 owns held-out confirmation and amortization.

## Acceptance criteria

- [ ] Comparators A–D share frozen data, teacher snapshots, training compute, model-capacity band,
      selected Field V2/codec version, byte target, seeds, and refinement recipe; incompatibilities
      in A are reported rather than repaired silently.
- [ ] Shuffling predicted or teacher rows leaves the set objective invariant within tolerance;
      rendered-pixel-equivalent fixtures are not forced apart by arbitrary row identity.
- [ ] Whole frames/captures and cameras are split before teacher generation. Adjacent views/frames
      cannot cross train/development boundaries; source and teacher provenance is hash-bound.
- [ ] The POD loop is reproducible: predict, run a frozen short refinement, detach the selected
      checkpoint, and refresh targets on a recorded cadence without gradient through the fitter.
- [ ] Development report includes `0/50/200/500` refinement PSNR/MS-SSIM/LPIPS, BENCH-019
      downstream objective, complete bytes, time-to-target/AUC, inference/refinement latency,
      teacher/training cost, memory, prediction survival, and failures.
- [ ] A stepwise killing rule isolates B−A locality/set value, C−B render value, and D−C POD value;
      one candidate advances to FF-003 or the learned branch stops.
- [ ] NumPy/torch import split, deterministic top-N, focused tests, portable report/audit, ARA
      disposition, docs/task synchronization, and `./scripts/verify.sh` pass.

## Interfaces touched

`src/structsplat/predictor.py`, Field V2/codec and short-fit adapters, teacher/training/experiment
drivers under `scripts/experiments/`, tests/report artifacts, `docs/additive_field_v2.md`, this task,
and the Index.

## Depends on

FF-001, CORE-013, BENCH-020/021/025, COMP-013/014, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`.

## Notes

This task supplies a learned candidate, not a generalization claim. The definitive fast-tier and
training break-even decision is BENCH-023 after the iterative production profile is confirmed.
