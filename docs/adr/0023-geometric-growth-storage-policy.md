# ADR-0023: Geometric-growth storage policy

- Status: accepted (interfaces); opt-in, default off
- Date: 2026-07-24
- Task: FIT-026
- Related: ADR-0020, ADR-0021, ADR-0022, FIT-021, FIT-023, FIT-024, FIT-025

## Context

ADR-0022 separated physical capacity from the logical activation ceiling, but for
`storage_policy="fixed_capacity"` the physical tensor is still pre-allocated once at `capacity`.
A fixed capacity forces a single up-front decision: size it to the byte budget and every fit pays
for parked rows it may never activate, or size it tight and the fit hard-stops at the ceiling.

The realtime-gs `GeometricParameterArena` (its Stage-3 storage) instead amortizes physical
allocation by growing capacity geometrically as the active set grows. StructSplat's transactional
prefix (ADR-0021) plus the active-prefix Adam optimizer (`_ActivePrefixAdam`) already provide the
two pieces that arena relies on — a contiguous active prefix and capacity-sized moments updated only
over that prefix — so the same amortization is available here without a renderer or optimizer
change. Physical allocation and logical activation stay separate decisions (ADR-0022).

## Decision

1. Add `storage_policy="geometric"`. The physical row count (`field.n`) starts at
   `initial_capacity` (default: the initial field size) and grows geometrically by `growth_factor`
   (default `2.0`) toward `capacity`, which becomes the physical *maximum*. `base_active_limit`
   remains the logical ceiling of ADR-0022.
2. Growth is committed just before an ordinary topology auction, when the active prefix comes within
   one block's birth demand of the current physical capacity. Field tensors append parked rows;
   Adam moments grow through `adapt_optimizer_state` with the appended rows zeroed. Because each
   step at least multiplies by `growth_factor`, a fit ending at `N` rows migrates `O(log N)` times.
3. The live contiguous prefix is preserved bit-for-bit across a migration, and the active-prefix
   optimizer only touches the logical prefix. A geometric fit's accepted trajectory and its
   output-compacted field are therefore identical to the equivalent `fixed_capacity` run — growth
   only adds inactive storage. This is asserted at `atol=0` in the tests.
4. Births clamp to the physical headroom (`field.n`) rather than the configured ceiling. For
   `fixed_capacity`, `field.n == capacity`, so that path is unchanged; for `dynamic`, storage
   appends and the configured ceiling still bounds it.
5. Geometric storage does not support the detail-tail reserve, which remains `fixed_capacity`-only
   (ADR-0022): a pre-reserved inactive suffix and on-demand physical growth are mutually exclusive.
6. Storage telemetry reports `policy`, `growth_factor`, `initial_capacity`, the migration count, and
   the per-migration `{old_capacity, new_capacity, active_n}` events.
7. Defaults are unchanged (`storage_policy="dynamic"`). Geometric growth is opt-in; this ADR makes no
   default-method or fitting-quality claim.

## Consequences

- Physical allocation tracks the realized working set: a fit that ends well below its byte-budget
  ceiling never materializes or adapts the unused rows, while a fit that needs the ceiling still
  reaches it. The single up-front capacity guess of `fixed_capacity` is replaced by an amortized
  schedule.
- Because growth is a pure storage change that preserves the prefix, it cannot move fitting quality
  relative to `fixed_capacity` — unlike an active-count change, which does. The equivalence test
  guards this.
- This ADR establishes the mechanism and the fixed-capacity equivalence, not a timing result.
  Whether geometric allocation is faster or more memory-frugal than fixed or dynamic allocation on a
  real workload is a FIT-024-style benchmark question and is left open.
