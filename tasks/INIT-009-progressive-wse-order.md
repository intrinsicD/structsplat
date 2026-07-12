# INIT-009: Progressive WSE survivor ordering

## Status

Implemented and confirmed on the preregistered structural audit. The terminal set and default row
order remain unchanged; `wse_progressive_order=True` opts pure-WSE layouts into the repaired
prefix sequence.

## Context

`docs/theory.md` states that every prefix of the WSE layout is a valid blue-noise set. The current
`sampling.eliminate()` returns `np.nonzero(alive)[0]`, which sorts the terminal survivors by input
candidate index. Candidate-index prefixes are accidental and do not implement Yuksel's progressive
weighted sample elimination ordering.

The terminal exact-N survivor set is already validated and must remain unchanged. The missing
operation is to continue the same deterministic greedy elimination within that set, record the
removal history, and reverse it so every prefix is itself a survivor set at a lower count.

## Goal

Add an opt-in progressive ordering to `sampling.eliminate` and `InitConfig`, wire it into pure-WSE
initialization without changing the default, and prove that it preserves the terminal Gaussian set
while improving low-count prefix spacing/coverage over historical candidate-index order.

## Acceptance criteria

- [x] `eliminate(..., progressive=True)` returns the exact same terminal survivor set as the legacy
      call, differing only in order; `progressive=False` preserves the public API's old ordering.
- [x] The progressive order is deterministic and implements Yuksel's recursive halving and
      reverse-shell order for both Euclidean and anisotropic metrics.
- [x] Pure-WSE init paths can request progressive order and keep count, complete Gaussian tuples,
      spacing metadata, and feature-scale metadata aligned.
- [x] Tests cover no-reduction, one-sample, disconnected/tied graphs, density-varying radii, and
      anisotropic metrics.
- [x] A reproducible prefix audit compares candidate-index and progressive prefixes at
      N/16, N/8, N/4, and N/2 using minimum spacing and maximum candidate coverage hole.
- [x] Documentation distinguishes within-WSE progressive order from the coarser pyramid-level
      append order and calibrates the prior-art claim to Yuksel 2015.

## Preregistered evidence and decision rule

Use eight deterministic uniform candidate sets with M=2048 and terminal N=256. For every seed,
compute one terminal WSE set, compare the historical sorted order with the progressive order, and
evaluate prefixes {16, 32, 64, 128}. The repair is accepted if:

- terminal survivor sets match exactly in every seed;
- progressive ordering improves mean normalized minimum spacing and mean inverse coverage-hole at
  every prefix size; and
- it wins both raw metrics on at least 75% of the 32 seed/prefix pairs, with ordering overhead no
  more than 25% of terminal-set WSE time.

If the ordering fails the quality gate, correct `docs/theory.md` to scope the LOD claim only to
pyramid-level prefixes and keep the legacy public behavior. Do not tune the metric on this audit.

## Evidence and decision (2026-07-13)

Command:

```bash
python -m benchmarks.wse_prefix_audit \
  --outdir results/init009_wse_prefix_audit \
  --seeds 0 1 2 3 4 5 6 7 --candidates 2048 --terminal 256 \
  --prefixes 16 32 64 128
```

Committed artifact: `ara/evidence/init009-wse-prefix-audit-2026-07-13/`, source-bound to clean
commit `916245e000499179739d5d3438b77062c7d27a9e`.

The terminal survivor set matched in all eight uniform Euclidean seeds. Progressive ordering won
both normalized minimum spacing and inverse normalized coverage-hole in all 32 descriptive
seed/prefix pairs (four correlated prefixes per independently generated terminal set):

| Prefix | Spacing gain | Inverse-hole gain |
|---:|---:|---:|
| 16 | +0.6599 | +0.2770 |
| 32 | +0.5141 | +0.4401 |
| 64 | +0.3908 | +0.4077 |
| 128 | +0.2153 | +0.3210 |

The ordering subroutine took 14.2% of this uniform terminal-set WSE selection time, below the 25%
gate; this is not an end-to-end anisotropic/quadtree initialization-overhead claim. Accept the
opt-in repair. Keep the compatibility default off because row order affects NPZ hashes, predictor slots,
CUDA atomic summation, and resumed artifacts even though complete Gaussian tuples are identical.
The current codec Morton-sorts fields, so this result does not establish progressive bitstreams.
Algorithm tests cover variable radii and anisotropic recursive semantics, but the quality audit does
not establish 32/32 gains for those recipients or optimality over alternative progressive orders.

## Interfaces touched

`src/structsplat/config.py`, `src/structsplat/sampling.py`, `src/structsplat/init.py`,
`src/structsplat/cli.py`, `benchmarks/wse_prefix_audit.py`,
`tests/test_sampling.py`, `tests/test_init_stages.py`, `tests/test_cli.py`, `docs/theory.md`,
`docs/architecture.md`, `benchmarks/README.md`.

## Depends on

INIT-003, INIT-005, INIT-006, BENCH-002.

## Research provenance

Selected as portfolio candidate P3 after FIT-017 failed its preregistered guard. Yuksel's weighted
sample elimination already supplies the progressive-order principle; StructSplat's contribution
here is an anisotropic/density-adaptive correctness repair and evidence, not algorithmic novelty.
