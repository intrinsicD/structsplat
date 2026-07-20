# ADR-0016: Keep marginal cold-stream birth benchmark-only

## Context

COMP-006 tested whether one residual-placed standard Gaussian is a better use of a complete SSPL1
stream allowance than count-neutral replacement or global precision reallocation. From one
cold-decoded N=64 parent per target and seed, it compared 16 matched births, 16 replacements, and
an exhaustive 875-point precision grid at an integer cap of 16 bytes above the matched no-edit
stream. Every candidate was encoded, persisted, cold-decoded, and centrally scored.

The frozen development screen was decisive. All 36 cells were feasible, but birth lost
`-1.0714 dB` mean and `-0.9533 dB` median paired PSNR to the strongest control. The
family-stratified bootstrap 95% interval was `[-1.2873, -0.8417] dB`; all six family means were
negative. A same-source replay matched all non-timing evidence exactly. Precision reallocation was
the strongest control in 23/36 cells and count-neutral replacement in 13/36.

Complete-stream accounting still exposed a proxy failure: the actual-byte and nominal-raw-bit
oracles selected the same row in only 14/36 cells. This supports retaining exact byte accounting as
benchmark infrastructure, but it does not establish an additive local byte price or a deployable
selector. Broad action class still agreed in 34/36 cells, and standard birth was the exact winner
in only 5/36.

## Decision

Do not add the COMP-006 birth oracle, marginal-byte score, or a rate-aware structural-birth policy
to production. Stop this frozen standard-birth/cold-SSPL1 formulation and do not consume its odd-
variant confirmation split. Keep complete-stream counterfactual selection as benchmark
infrastructure for materially different codec or primitive hypotheses.

## Consequences

+ `GaussianField`, the allocator, fitter, renderer, codec syntax, CLI, configuration, and defaults
  remain unchanged.
+ The benchmark supplies reusable cold-stream validation, component accounting, integer-cap
  selection, source snapshots, and exact replay checks.
+ Gaussian count and analytical payload are rejected as sufficient rate proxies; future
  compression claims must still count complete self-contained streams.
+ A richer-atom test requires a real versioned codec syntax and equal-byte controls. Affine color,
  luminance slope, and WIPES-like carriers cannot be credited with proxy bytes.
+ Exact per-Gaussian backward remains an independent performance lane and inherits no speed claim
  from this CPU encoding audit.
- No confirmation, natural-image claim, local additive price model, production selector, or
  compression/expressiveness improvement is authorized by COMP-006.

## Links

Follows ADR-0015 and depends on COMP-006, COMP-001/002/004, BENCH-007, and FIT-017. It does not
supersede a shipped architecture decision because no marginal-rate allocator entered production.
