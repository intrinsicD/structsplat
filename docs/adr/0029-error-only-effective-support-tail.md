# ADR-0029: Optional error-only effective-support tail

- Status: accepted (interface); one exposed-image screen completed, method remains default-off
- Date: 2026-07-27
- Task: FIT-031
- Related: ADR-0022, ADR-0025, ADR-0028, FIT-025, FIT-030

## Context

The current safe schedule can stop with visible fine residual. FIT-025 already tested a
fixed-capacity, fixed-512-row detail reserve that selected covered deep-interior high-frequency
sites and auctioned births against splits. On its one-image development screen, ordinary
activation was better at equal count, so that specialized tail correctly remained off.

The requested follow-up asks a different question: after the complete ordinary schedule, estimate
how spatially broad the remaining error is, allocate half of that estimated demand as genuinely
small residual-ranked Gaussians, and continue optimizing until the safe schedule reaches a fixed
point. The estimate must be explicit because “enough rows to completely remove the error” is not
identifiable from one rendered field and cannot be promised by a normalized Gaussian model.

## Decision

1. `scripts/convert.py --fine-detail` enables one terminal stage after `safe_polish`. Omitting the
   flag leaves the current recipe and its 11,000-row ordinary capacity unchanged.
2. Let `e` be foreground per-pixel RGB mean absolute error after the ordinary schedule. Estimate
   complete residual demand with the effective-support participation ratio
   `N_eff = ceil((sum e)^2 / sum(e^2))`. Request `ceil(0.5 * N_eff)` rows. The estimate, its
   components, and the requested fraction are persisted; this is a heuristic, not a zero-error
   guarantee.
3. Candidate ranking uses only `e`. There is no target structure tensor, target-frequency prior,
   coverage deficit, or coherence term. New rows are isotropic, use a residual-support footprint
   capped at `1.25` px, and take their initial color from the target pixel. The mask remains a
   geometric feasibility constraint and the existing certified containment projection/caps apply.
4. Dynamic storage may grow beyond the ordinary capacity only inside this opt-in tail. Rows enter
   in batches of at most 512 and are not bisected below the schedule's existing
   `event_min_count` (eight in the current profile). Each batch recovery-fits its touched rows and
   must pass the unchanged full protected-metric Pareto gate; rejected batches are geometrically
   reduced.
5. Accepted topology is followed by global low-learning-rate L2 optimization in 250-step blocks.
   It stops at the deterministic safe fixed point or a logged 4,000-step ceiling. The ceiling makes
   “until convergence” operational and prevents an unbounded command.
6. Config, transition history, storage telemetry, result rows, and `index.html` report the
   estimate, requested/activated rows, allocation and convergence reasons, and protected metrics
   before and after the stage.

## Consequences

- The option can spend substantially more rows and wall-clock than the default; its output is not
  rate-matched to the ordinary recipe.
- The method tests whether explicit fine capacity improves visible residual, not whether it is an
  efficient codec or a new default.
- FIT-025's negative equal-count result is preserved because this tail changes the estimator,
  placement geometry, count regime, storage timing, and terminal position.
- This does not implement FIT-030: there is no residual EMA, continuous birth/death policy,
  per-row bit price, `D + lambda R`, or removal of phase boundaries. FIT-030 remains blocked on its
  preregistered precursors.

## Development screen

The 2026-07-27 masked Janelle C0001 seed-0/max-side-1200 screen estimated 14,177 effective sites,
requested 7,089 rows, and committed nine 512-row batches (4,608 rows). The tenth wave failed every
gate replay from 512 through eight rows. Low-learning-rate convergence then rejected its first
backtracked block and logged the deterministic fixed point.

Within that run, foreground/boundary PSNR improved `+0.522239/+0.582752 dB`, CVaR99/p99 MSE fell
`12.09%/14.82%`, and exact-zero interior/outside metrics were preserved. This is a source-bound
effect of the full tail stage on one exposed development image, not an equal-count or rate result.
The existing clean default stopped at 10,824 rows while this run reached 11,000 before the tail,
and CUDA trajectories are not bit-exact. C58 therefore authorizes no default, generality,
efficiency, or codec claim. Evidence:
`ara/evidence/fit031-error-only-tail-janelle-2026-07-27/`.

## Links

Realizes FIT-031 through `safe_schedule`, `pipeline`, and the sole conversion CLI established by
ADR-0028. C57 binds the implementation-only capability; any quality interpretation needs a
separate source/config-bound experiment and audit.
