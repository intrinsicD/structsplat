# BENCH-018: How much should the commit gate cost? (`block_steps` granularity)

## Status

Todo. Needs GPU; no code change required — `--block-steps` is already exposed.

## Context

The transactional gate trials a whole block of `block_steps` optimization steps and rolls it back
if the metric vector regresses. `block_steps` therefore sets the **unit of discarded work**: at the
schedule default of 250, one rejection throws away up to 250 steps.

The BENCH-017 exploratory pass measured the resulting acceptance rates on the full-frame arm:

| phase | attempted | accepted | acceptance |
|---|---:|---:|---:|
| bootstrap | 7,776 | 3,200 | 41.2% |
| coverage_growth | 3,034 | 420 | 13.8% |
| detail_growth | 61,028 | 3,760 | 6.2% |
| redistribution | 26,536 | 1,840 | 6.9% |
| safe_polish | 3,276 | 0 | 0.0% |

`detail_growth` alone is ~56% of wall-clock at 6.2% acceptance. Overall, ~9% of attempted steps
commit.

This is the direct trade-off question: a smaller block discards less per rejection but evaluates
the (expensive) gate metric more often. Nobody has measured where the optimum is, and the schedule
default of 250 was inherited, not chosen against data.

## Goal

Find the `block_steps` that maximizes terminal quality per unit wall-clock, and decide whether the
schedule default should change.

## Approach (preregister before running)

1. **Arms**: `block_steps` in `{25, 50, 100, 250, 500}` at the shipped recipe, everything else
   fixed. Full-frame on Kodak-24 and masked on the dome fixture, >= 3 seeds.
2. **Equal wall-clock envelope is the point** — do not equalize step counts. Report terminal
   quality against *time*, since a smaller block buys fewer wasted steps at the price of more gate
   evaluations.
3. **Report per cell**: PSNR / MS-SSIM / LPIPS, attempted vs accepted steps, gate evaluations,
   fraction of wall-clock spent inside the gate metric, and the rejection-reason histogram.
4. **Frozen gate**: change the default only on a paired win in quality-at-equal-time with a 95% CI
   excluding zero, on both arms.

## Acceptance criteria

- [x] Preregistered protocol, arms, and gate committed before any target is fitted. *(Masked arm,
      2026-08-08; the full-frame Kodak-24 arm is not preregistered and has not run.)*
- [x] Per-cell config/provenance per BENCH-002. *(15/15 cells, no error cells; bundle passes
      `check_report_bundle.py --allow-dirty` with no config-versus-manifest divergence.)*
- [x] Gate evaluation cost isolated and reported, not inferred from total time. *(Per-phase
      attempted/accepted steps, block counts, and rejection histograms via `gate_telemetry`.)*
- [x] Decision recorded in `ara/logic/claims.md`; schedule default changed or explicitly kept.
      *(**Kept at 250**; `C66`.)*
- [ ] Full-frame Kodak-24 arm preregistered, executed, and gated; distinct review obtained before any
      default change.

## Frozen masked-arm development protocol (2026-08-08)

This freezes **only the masked arm**; the full-frame Kodak-24 arm and the promotion gate above are
untouched. No distinct prospective reviewer exists for this run, so it is a **development
diagnostic** that can inform the default but cannot change it alone.

- **Question:** where is the optimum of the discarded-work versus gate-cost trade-off, measured as
  terminal quality per unit wall-clock rather than per step?
- **Data / regime / hardware:** identical to FIT-028's frozen masked-arm protocol — masked Janelle
  `frame_00008/C0001` at `--max-side 1200`, shipped recipe, capacity 11,000, RTX 3050.
- **Arms:** stage `commit_gate`, variants `current` (the inherited 250), `block25`, `block50`,
  `block100`, `block500`. One granularity is applied to every gated phase and clamped to that
  phase's ceiling, exactly as `PipelineConfig.block_steps` does; a test asserts `max_steps` and
  `target_gaussians` are unchanged.
- **Seeds:** 0, 1, 2, pairing on (image, seed).
- **Equal wall-clock is the point.** Do not compare at equal steps. Report terminal PSNR/MS-SSIM/
  LPIPS against `fit_seconds`, plus attempted/accepted steps, block counts, and the rejection
  histogram from `gate_telemetry`.
- **Reading rule, declared before outcomes:** a smaller block is only interesting if it wins
  terminal quality *at no more wall-clock* than `current`. A higher acceptance rate at greater
  total time is the expected mechanical consequence of evaluating the gate more often and is not
  by itself evidence for changing the default.
- **Known cost calibration (pre-run, `--max-side 256`, seed 0, diagnostic):** `current` fits in
  200.5 s at 8.01% step acceptance; `block100` fits in 464.2 s at 10.91%. Acceptance rises and
  wall-clock rises faster. This calibration is why the reading rule prices time explicitly.
- **Exact command:**

```bash
python scripts/stage_search.py IMAGES OUTDIR --mask-dir MASKS \
  --stage commit_gate --seeds 0 1 2 --max-side 1200 --lpips
```

- **Forbidden follow-ups:** changing tolerances or phase ceilings to make a block size look better;
  promoting a schedule default from the masked arm alone.

## Depends on

BENCH-002, BENCH-017, FIT-023, ADR-0025

## Notes

Interacts with FIT-028: relaxing the interior-hole veto changes the rejection rate, which changes
the optimal block size. Run FIT-028 first, or run the two as a factorial if the budget allows —
but do not read a `block_steps` optimum measured under the strict veto as valid after the veto
changes.

Also interacts with FIT-027: the gate metric is dominated by SSIM, so making SSIM cheaper moves
the optimum toward smaller blocks. Measure after FIT-027 lands.

## Agent workflow

- Driver: claude-root
- Reviewer: pending-distinct
- Turn: driver
- Reviewed revision: pending

### Handoff log

The masked-arm protocol above was frozen on 2026-08-08 before any target was fitted, but no
distinct prospective reviewer was available to approve its digest. The run is therefore a
development diagnostic under `structsplat-experiment`'s first classification: it can inform
pipeline design and it can kill a knob, but it cannot promote a default or close the promotion
gate, which still requires the full-frame Kodak-24 arm and distinct review.

## Masked-arm outcome — 2026-08-08: default kept at 250

| arm | PSNR | ΔPSNR [95% CI] | acceptance | reached 11,000 |
|---|---:|---|---:|---:|
| `block25` | 26.4027 | +0.317 [-0.219, +0.852] | 16.52% | 3/3 |
| `block50` | **26.5292** | **+0.443 [+0.111, +0.775]** | 14.33% | 3/3 |
| `block100` | 25.9185 | -0.168 [-0.598, +0.263] | 9.65% | 1/3 |
| `current` | 26.0860 | — | 8.77% | 1/3 |
| `block500` | 26.1064 | +0.020 [-0.264, +0.305] | 8.18% | 0/3 |

**No default change.** The frozen gate requires the paired quality win on *both* arms and the
full-frame arm has not run, so `block50`'s significant masked PSNR win cannot promote anything. It
also fails multiplicity: four comparisons are nominally significant, Bonferroni over the 20
comparisons at n=3 needs `|t| ~ 28`, and they reach `5.8 / 13.3 / 25.8 / 6.9`. Different arms clear
different responses, which is the signature of repeated noise sampling.

**The finding is the non-proxy result.** Acceptance is monotonic across the full 20x range and
capacity attainment tracks it, while terminal PSNR is monotonic in neither. With FIT-028's inert
budget ladder, two independent knobs cleanly control the gate's accept rate and neither converts it
into image quality (`C66`). The actionable defect is capacity: coarse blocks systematically fail to
deliver the requested budget, and `block500` misses by ~18% in every seed.

**Protocol correction.** The pre-run calibration in this task (`--max-side 256`: `current` 200.5 s
versus `block100` 464.2 s) predicted that smaller blocks cost more wall-clock. At the production
resolution the sign reverses — `block25` fits faster than `current` in all three seeds — because the
calibration was run at a scale where the gate metric dominates rather than the optimization. The
time-pricing reading rule it motivated stands; its directional claim does not.

**Design constraint discovered here.** The grid re-ran FIT-028's baseline configuration, giving three
same-config replicate pairs: PSNR sd `0.185 dB`, identical target pixels, differing fields. Seed does
not pin the trajectory, the detection floor at n=3 is `~0.46 dB`, and resolving `0.1 dB` would need
15--20 seeds (`C67`). This bounds what any future n=3 screen of this pipeline can conclude, including
the full-frame arm still owed here.

Evidence `ara/evidence/bench018-commit-gate-janelle-2026-08-08/run.md`; claims `C66`/`C67`; trace
`N266`/`N267`/`N268`; staging `O140`/`O141`.
