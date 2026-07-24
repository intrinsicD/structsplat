# ADR-0022: Separate physical capacity from staged activation

- Status: accepted (interfaces); specialized detail tail remains experimental
- Date: 2026-07-24
- Task: FIT-025
- Related: ADR-0020, ADR-0021, FIT-021, FIT-023, FIT-024, CORE-010/011

## Context

ADR-0021 made fixed-capacity storage independent of topology policy, but
`SafeScheduleConfig.capacity` still acted as both the physical tensor size and the logical row
ceiling. Increasing preallocated storage therefore exposed every additional row to ordinary
coverage, boundary, and redistribution growth. It was impossible to hold physical headroom for a
late refinement experiment without changing the earlier topology trajectory.

Physical allocation, ordinary activation, and late-stage activation are separate decisions.
Conflating them also makes a storage-only change appear to change fitting quality.

## Decision

1. `capacity` is the physical row count for `storage_policy="fixed_capacity"`.
   `base_active_limit` is the logical ceiling visible to the ordinary coverage, detail, boundary,
   and redistribution phases. `None` resolves to `capacity`, preserving historical behavior.
2. `detail_tail_max_rows` reserves an optional suffix after `base_active_limit`. A nonzero reserve
   requires fixed-capacity storage and must fit within physical capacity. Inactive rows remain
   outside rendering, projection, loss, and Adam updates.
3. The tail runs after the pre-polish color solve. It ranks persistent signed high-frequency
   residuals only at already-covered deep-interior pixels, auctions detail births against
   moment-preserving splits, recovery-fits only touched rows, and applies the unchanged
   full-resolution Pareto gate.
4. Tail activation is transactional in batches. A batch may be halved after rejection, and the
   phase stops at the first auction with no safe and sufficiently effective winner.
   `detail_tail_min_gain_per_row` provides an opt-in effectiveness floor; zero retains the ordinary
   material-gain gate.
5. Config, transition history, and storage telemetry report physical capacity, base active limit,
   configured tail reserve, batch size, termination reason, and realized activated rows.
6. Defaults remain `base_active_limit=None` and `detail_tail_max_rows=0`. Neither fixed storage nor
   the detail-tail policy is promoted globally by the single-image FIT-025 screen.

## Consequences

- A physical pool can contain unused headroom without changing the active field or ordinary
  topology policy.
- Increasing active count remains a model-capacity change and can affect fitting quality and
  runtime; increasing only inactive physical capacity is a storage change.
- The FIT-025 development screen found that a generic +512 active budget strictly outperformed the
  specialized +512 tail on every nontrivial protected metric and was faster in the observed run.
  The detail tail therefore remains an opt-in research mechanism, not the recommended policy.
- All three FIT-025 arms used identical fixed physical storage, so their timing does not measure
  fixed versus dynamic allocation. FIT-024 remains the relevant storage-policy timing evidence.
- A gain-per-row stopping floor requires a predeclared replicated experiment. The descending
  gains observed in FIT-025 may motivate such a test but do not authorize a post-hoc default.
