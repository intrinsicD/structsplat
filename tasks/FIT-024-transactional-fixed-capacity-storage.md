# FIT-024: Transactional fixed-capacity storage

**Status: implemented (opt-in); focused parity and one-image A/A-calibrated CUDA check complete.**

## Context

FIT-023's safe schedule grew `GaussianField` and Adam row tensors for every birth/split proposal.
FIT-021 already proved that parked off-image rows can make a fixed-capacity allocation inert, but
its pool was coupled to a different `park → merge → split → spawn` triage policy and could not use
state-matched checkpoints. Storage layout should not select topology policy.

## Decision

Add `SafeScheduleConfig.storage_policy` with:

- `dynamic`: the historical append/pad path and default;
- `fixed_capacity`: capacity-shaped accepted/trial fields, an immutable contiguous `active_n`
  prefix per state, fixed-shape Adam moments, no growth-time append/pad resize, and one terminal
  compaction. Detached proposals/checkpoints still allocate transaction scratch.

The fixed path keeps FIT-023's proposal factories, identical-state auction, recovery fitting,
full-resolution commit vector, batch backtracking, and Pareto field+Adam checkpoints. Births and
splits write into `[active_n:new_active_n)` and zero those rows' moments. Count-neutral operators
edit existing active rows. Rejected trials discard the trial field, optimizer state, and proposed
active count together.

`fit(active_row_count=...)` keeps full-capacity leaf tensors in the optimizer while render/loss
functions consume a zero-copy prefix view. Adam moments retain capacity-sized storage, while
projection, clamping, frozen-row handling, and Adam update kernels operate at the active shape.
Inactive rows are frozen and exempt from mask projection. Dynamic topology inside such a fit
block is rejected.

## Acceptance criteria

- [x] Capacity shape is established before the first optimizer block; growth never resizes field
      or Adam row tensors.
- [x] Growth proposals keep physical tensor shapes fixed and advance only trial-local `active_n`.
- [x] Touched Adam rows are reset without padding/reallocation; Adam kernels update only the
      active prefix.
- [x] State-matched Pareto checkpoints work with fixed-capacity field and Adam snapshots.
- [x] Local/global recovery masks exclude inactive rows.
- [x] Observers and final consumers see active rows only; final field and Adam state compact once.
- [x] CLI/config/history expose the storage policy and capacity lifecycle.
- [x] Focused tests cover shared-storage prefix autograd, immutable activation, compact-fit
      numerical parity, proposal parity, and an end-to-end dynamic/fixed schedule trajectory.
- [x] Run the winning Janelle arm as dynamic A → fixed capacity → dynamic B from one source state,
      audit the full commit histories, and report phase plus end-to-end timings.

## Evidence boundary

The focused CPU fit differs only at floating-point noise scale in a multi-step Adam test, and the
tiny end-to-end schedule selects identical events, metrics, and final tensors.

The full source-bound evidence is
`runs/janelle_C0001_storage_ab_active_shape_20260724/`:

| arm | FG PSNR | boundary PSNR | schedule seconds | total seconds | peak GPU MiB |
|---|---:|---:|---:|---:|---:|
| dynamic A | 27.0670 | 11.4206 | 271.90 | 407.03 | 2142.1 |
| fixed capacity | 27.0629 | 11.3998 | 272.12 | 405.50 | 2130.2 |
| dynamic B | 27.0629 | 11.3945 | 285.19 | 418.43 | 2141.5 |

Fixed capacity is inside the dynamic A/A envelope for terminal quality (foreground is 0.000003 dB
below the lower dynamic endpoint; boundary lies between the controls), saves about 11.5 MiB peak,
and is runtime-neutral within path variance. Per attempted step it also lies between the dynamic
controls. Repeated renders of one CUDA field differ by one ULP, and the two dynamic controls chose
different accepted proposal paths, so bitwise/event-history identity is not a valid criterion.

This is one image, one seed, and one device. It supports “no detected fitting-quality effect,” not
universal equivalence or a meaningful speed claim. `dynamic` remains the default until multi-image
evidence justifies promotion.

The adversarial claim/provenance audit is
`ara/evidence/fit024-transactional-fixed-capacity-janelle-2026-07-24/run.md` and ARA claim C51.

## Interfaces touched

`src/structsplat/{gaussians,pool,fit,safe_schedule}.py`,
`deprecated_scripts/fit_janelle_safe_commit_schedule.py`, focused tests, README/architecture, and ADR-0021.

## Depends on

FIT-021, FIT-023, CORE-010/011, ADR-0020.
