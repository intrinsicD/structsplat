# BENCH-018 commit-gate granularity — masked Janelle development screen

## Scope

Masked-arm half of BENCH-018, plus a same-config replication study that the grid produced as a
by-product. One exposed development image, three seeds, one GPU, no distinct prospective reviewer:
a **development diagnostic**. The full-frame Kodak-24 arm has not run, so BENCH-018's frozen gate —
which requires the win on *both* arms — cannot be satisfied here and **no default changes**.

## Protocol

Frozen in `tasks/BENCH-018-commit-gate-granularity.md` before the first fit, including the
time-pricing reading rule.

- **Source/regime/hardware:** identical to FIT-028's masked arm — Janelle
  `frame_00008/rgb/C0001.jpg` (SHA-256 `ae24fe99…`) with `mask/mask_C0001.png` (SHA-256
  `94dcbf70…`), `--max-side 1200`, shipped recipe, capacity 11,000, `quadtree_wse`, exact CUDA
  renderer, seeds 0/1/2, RTX 3050.
- **Arms:** the `commit_gate` stage — `current` (the inherited 250), `block25`, `block50`,
  `block100`, `block500`. One granularity is applied to every gated phase and clamped to that
  phase's ceiling; a focused test asserts `max_steps` and `target_gaussians` are unchanged.
- **Reading rule:** compare at wall-clock, not at equal steps; a higher acceptance rate at greater
  total time is a mechanical consequence of evaluating the gate more often and is not by itself
  evidence for changing the default.

## Result

Means over three seeds; paired deltas against `current` with 95% Student-t intervals at n=3.

| arm | PSNR | ΔPSNR [95% CI] | fit s (per seed) | acceptance | attempted | reached 11,000 |
|---|---:|---|---|---:|---:|---:|
| `block25` | 26.4027 | +0.3167 [-0.219, +0.852] | 423 / 573 / 374 | 16.52% | 14,268 | 3/3 |
| `block50` | **26.5292** | **+0.4432 [+0.111, +0.775]** | 467 / 448 / 718 | 14.33% | 15,941 | 3/3 |
| `block100` | 25.9185 | -0.1675 [-0.598, +0.263] | 491 / 825 / 577 | 9.65% | 18,620 | 1/3 |
| `current` (250) | 26.0860 | — | 679 / 665 / 675 | 8.77% | 18,856 | 1/3 |
| `block500` | 26.1064 | +0.0204 [-0.264, +0.305] | 607 / 546 / 560 | 8.18% | 16,799 | 0/3 |

Four comparisons are nominally significant in isolation: `block50` PSNR (`+0.4432`), `block25`
LPIPS (`-0.0017`), `block100` LPIPS (`+0.0009`, i.e. worse), and `block500` fit seconds
(`-101.9`). **None survives multiplicity.** There are 20 comparisons (4 arms x 5 responses);
Bonferroni at n=3 requires `|t| ~ 28`, and these reach `5.8`, `13.3`, `25.8`, and `6.9`.

Note the shape of which arms clear which responses: `block25` clears LPIPS but not PSNR, `block50`
clears PSNR but not LPIPS, `block100` clears LPIPS in the *wrong direction*, `block500` clears only
time. That is the signature of repeatedly sampling noise, not of a mechanism asserting itself.

## The finding: gate acceptance is not a proxy for quality

Two responses are monotonic in block size across the full 20x range, robustly and without overlap
at the extremes:

- **Step acceptance**: `16.52 -> 14.33 -> 9.65 -> 8.77 -> 8.18%` as the block grows 25 -> 500.
- **Capacity attainment**: `3/3, 3/3, 1/3, 1/3, 0/3` cells reaching the requested 11,000 rows.
  `block500` misses by roughly 18% in every seed (`8,968 / 9,096 / 8,968`), which is systematic
  rather than noisy.

**Terminal quality is monotonic in neither.** PSNR runs `26.403, 26.529, 25.918, 26.086, 26.106`:
`block50` is best, `block100` is worst, and `block500` is indistinguishable from the inherited 250.
No ordering of block sizes is consistent with both acceptance and quality.

This reproduces FIT-028's lesson from the opposite direction. There, the ADR-0026 budget moved
acceptance monotonically (`8.71 -> 10.48%`) and quality not at all. Here, block size moves
acceptance monotonically over a wider range and quality moves non-monotonically. **Two independent
knobs cleanly control the gate's accept rate; neither converts it into image quality.** For a
schedule whose design premise is that accepted work is good work, that is the substantive result,
and it rests on large monotonic mechanism effects rather than on sub-noise quality deltas.

The actionable consequence is capacity, not quality: coarse blocks systematically fail to deliver
the requested Gaussian budget, and `block500` fails it every time. Whether a run returns the
topology it was asked for is a correctness-shaped property, unlike a fractional-dB metric delta.

## Replication study (by-product)

The grid re-ran `current` at seeds 0/1/2 under the identical configuration FIT-028 had already
executed, giving three same-config replicate pairs. `target_pixel_sha256` matches in all three;
`field_sha256` matches in none.

| response | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---:|---:|---:|---:|---:|
| PSNR | +0.22497 | -0.14275 | +0.07714 | +0.053 | **0.185** |
| Gaussians | +944 | -48 | -384 | +171 | 690 |
| fit seconds | +88.3 | -47.0 | -23.3 | +6.0 | 72.2 |

The mean difference is indistinguishable from zero and the signs alternate, so the two grids are
exchangeable: this is symmetric run-to-run nondeterminism, not drift between runs.

**Seed does not pin the trajectory.** CUDA atomic accumulation is not bit-reproducible, and the
transactional gate makes threshold comparisons on metrics computed from those reductions, so a
ulp-level difference flips a borderline block from accept to reject, which changes topology, which
changes every later decision. The gate *amplifies* float nondeterminism into a 944-row divergence.

Consequences, which bound both screens:

- At n=3 the 95% t interval half-width is `4.303 x 0.185 / sqrt(3) ~ 0.46 dB`. That is the smallest
  PSNR effect either design can resolve, and it explains why every FIT-028 interval contained zero
  (largest estimate `0.212 dB`).
- Resolving a `0.1 dB` block-size difference needs roughly 15--20 seeds, not 3.
- No absolute metric from one grid is portable to another. Only within-grid paired comparisons are
  meaningful, and even those carry this envelope.

Caveat: n=3 pairs, one image, one regime. This bounds the envelope; it does not characterise it
across images, resolutions, or capacities.

## Integrity and limitations

- Bundle: `results/bench018_commit_gate_janelle_frame00008_2026-08-08/` (git-ignored);
  `manifest.json` SHA-256 `753402d1a7b14880c21050ef492ee44f17e9dd1e91eb0e1c37224ce50999f41d`.
  Manifest records commit `a8e8dde89a7560f4a13c356ebaf556a5d9b67437`, branch `main`, `dirty: true`,
  `status_sha256 ef8ab402…`.
- **`python scripts/check_report_bundle.py … --allow-dirty` passes**, with no config-versus-manifest
  divergence. The working tree was deliberately frozen for the whole run after FIT-028's bundle
  failed that check; `--allow-dirty` remains a disclosure, not a waiver.
- 15/15 cells completed; no error cells, no missing cells.
- A pre-run cost calibration recorded in the task at `--max-side 256` (`current` 200.5 s versus
  `block100` 464.2 s) predicted that **smaller blocks cost more wall-clock**. At the production
  resolution the sign reverses: `block25` fits faster than `current` in all three seeds. The
  calibration was run at a scale where the gate metric dominates; at `1200x1038` the optimization
  dominates. The time-pricing reading rule it motivated remains correct; its directional claim does
  not, and is corrected here.
- Same exposed image, capture group, seeds, GPU, and non-bit-reproducible CUDA as FIT-028. No
  held-out, default, generality, or compression claim follows.

## Reproduction

```bash
python scripts/stage_search.py /tmp/janelle/images \
  results/bench018_commit_gate_janelle_frame00008_2026-08-08 \
  --mask-dir /tmp/janelle/masks --stage commit_gate --seeds 0 1 2 --max-side 1200 --lpips

python scripts/experiments/fit028_bench018_gate_screen_report.py \
  results/bench018_commit_gate_janelle_frame00008_2026-08-08 --baseline current
```

Mask-tree staging is identical to `ara/evidence/fit028-hole-budget-janelle-2026-08-08/run.md`.
`index.html` carries per-cell curves, native-resolution target/reconstruction/error images, and
commit-gate accounting; `comparison.html` adds paired deltas, per-phase acceptance, both rejection
footings, and quality against wall-clock.
