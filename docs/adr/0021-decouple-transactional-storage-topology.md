# ADR-0021: Decouple transactional fixed-capacity storage from topology policy

- Status: accepted (opt-in, default off)
- Date: 2026-07-24
- Task: FIT-024
- Related: ADR-0020, FIT-021, FIT-023, CORE-003/010/011

## Context

FIT-021's fixed-capacity pool removed fit-loop tensor resizing, but enabling it also selected its
in-place triage algorithm. FIT-023's stronger source-bound schedule requires a different invariant:
all operators propose from one immutable accepted field, recovery-fit independently, and commit
only the best full-resolution Pareto-safe field+Adam state. Its birth/split implementation still
appended tensors and padded optimizer moments.

Preallocation is a storage decision. It must not silently choose triage order, donor policy,
acceptance metrics, or checkpoint behavior.

## Decision

1. `SafeScheduleConfig.storage_policy` selects `dynamic` or `fixed_capacity`; it does not select a
   topology operator.
2. Fixed storage establishes `schedule.capacity`-shaped accepted/trial tensors before the first
   optimizer block and never changes their row shape during growth. Logical state is an immutable
   contiguous active-prefix length. Safe-schedule growth is append-only, so an arbitrary mutable
   free list is unnecessary. Detached proposal and checkpoint snapshots still allocate
   capacity-shaped transaction scratch.
3. The optimizer owns full-capacity leaf tensors. `fit(active_row_count=N)` uses zero-copy prefix
   views for rendering and losses, freezes inactive rows/moments, and exempts them from mask
   projection. Adam moments remain capacity-sized, but projection, clamping, and Adam update
   kernels receive the active shape; inactive rows do not add optimizer work or a distinct
   capacity-shaped arithmetic path.
4. Growth proposals write reserved rows and reset their Adam moments. Count-neutral proposals edit
   active rows. Every auction candidate carries its own proposed active length, so rollback
   includes liveness as well as parameters and moments.
5. Pareto checkpoints retain fixed-shape field and Adam snapshots. Observers receive active views.
   The selected field and optimizer state compact once at the public output boundary.
6. Historical `dynamic` remains the default. Promotion requires broader evidence than one
   source-bound image because the CUDA renderer is last-bit nondeterministic and near-tied
   proposal auctions can therefore choose different safe trajectories even in dynamic A/A runs.

## Consequences

- Fixed storage can be tested against the historical algorithm without importing FIT-021 triage.
- Render/loss/Adam update work follows active count; field and Adam storage follow capacity.
- Repeated CUDA renders of an identical field differ at the last bit, and repeated dynamic runs
  can select different near-tied safe proposals. The contract is quality within an A/A-calibrated
  envelope with identical invariants and gates, not event-sequence or bit identity.
- `active_row_count` fit blocks reject internal prune/split/relocate/boundary-add/adaptive/triage
  events; topology remains owned by the transactional caller.
- Generation-density covariance filtering is not yet supported by this storage hook because its
  cohort variance must be assigned at logical birth count.
- The one-image A/A-calibrated Janelle check found no quality regression and a small peak-memory
  reduction, but no material runtime win. Transactional field/checkpoint clones and rendering
  dominate after append/padding reallocations disappear.
