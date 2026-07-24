# FIT-026 — Geometric-growth storage policy

## Context
Fixed-capacity storage (ADR-0020/0021/0022) pre-allocates the transactional pool once at
`SafeScheduleConfig.capacity`. That is a single up-front guess: too large wastes parked-row memory
and per-auction moment adaptation, too small hard-stops the fit at the ceiling. realtime-gs's
`GeometricParameterArena` amortizes this by growing physical capacity geometrically; StructSplat's
contiguous active prefix (ADR-0021) + `_ActivePrefixAdam` already hold the pieces to do the same.

## Goal
An opt-in `storage_policy="geometric"` for `run_safe_schedule` that starts at a small physical
capacity and grows it geometrically toward `capacity` on demand, without changing the fitted result
relative to `fixed_capacity` (growth adds only inactive storage).

## Acceptance criteria
- `pool.geometric_capacity_schedule` computes `min(cap, max(required, ceil(current * factor)))`
  (the realtime-gs arena schedule) and validates the factor/ceiling.
- `pool.grow_transactional_capacity` enlarges a field by appending parked rows, preserving the
  existing rows bit-for-bit; rejects shrink and `color_basis != "constant"`.
- `SafeScheduleConfig` gains `storage_policy="geometric"`, `growth_factor` (default 2.0), and
  `initial_capacity` (default = initial field size), with validation; detail-tail reserve stays
  `fixed_capacity`-only.
- `run_safe_schedule` grows physical storage + Adam moments (`adapt_optimizer_state`) before an
  auction when the prefix nears physical capacity, capped at the logical ceiling.
- A geometric fit is bit-identical (`atol=0`) to the equivalent `fixed_capacity` fit in field and
  metrics, while its storage telemetry shows ≥1 migration and a grown physical row count.
- `fixed_capacity` and `dynamic` runs are unchanged (existing safe-schedule tests stay green).

## Interfaces touched
`src/structsplat/pool.py` (schedule + grow primitive, parked-row helper),
`src/structsplat/safe_schedule.py` (`SafeScheduleConfig` fields/validation, setup, pre-auction
reserve, proposer physical-headroom clamp, storage telemetry), `tests/test_geometric_pool.py`,
`tests/test_safe_schedule.py`, `docs/adr/0023-geometric-growth-storage-policy.md`,
`docs/architecture.md`.

## Depends on
FIT-021 (pool lifecycle), FIT-023/FIT-025 (transactional safe schedule, ADR-0021/0022).

## Notes
- The bit-exact fixed-capacity equivalence is the safety property: geometric growth is a pure
  storage change and must not move fitting quality. It is enforced at `atol=0`.
- Timing/memory versus fixed and dynamic allocation is a FIT-024-style benchmark question, left
  open — this task establishes the mechanism and equivalence only.
- Ported as the StructSplat half of the realtime-gs `GeometricParameterArena` idea (the rtgs Stage-3
  `optim/arena.py`); StructSplat applies it to the Stage-1 transactional fitter instead.
- Developed on `claude/repos-preallocation-pool-strategy-yhdq9j` (a cross-repo infrastructure sweep).
