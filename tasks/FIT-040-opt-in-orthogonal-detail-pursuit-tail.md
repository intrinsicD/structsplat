# FIT-040: Opt-in orthogonal fine-detail pursuit tail

## Status

Implemented and screened on one exposed full-frame Janelle target. The separate current-pipeline
option reproduces FIT-039 and remains default-off; no recipe/default promotion is authorized.

## Context

On the persisted current-pipeline Janelle C0001 state, FIT-039's exact-site-only orthogonal
pursuit reaches the frozen large-effect target with 768 ordinary rows: deep sigma-1.5 high-pass
MSE falls `25.93%`, Laplacian MSE falls `27.32%`, LPIPS falls `10.46%` relative, and every
protected metric passes. Cold replay and fixed-point audit pass. FIT-031's committed 4,608-row screen used a
different crop, so that raw ratio is not comparative evidence; FIT-041 supplies the valid
same-base control and finds pursuit uses 3.616x fewer rows on the requested full frame.

## Goal

Make the mechanism reproducible through the sole current conversion entrypoint without changing
the existing default or FIT-031's `--fine-detail` behavior.

## Acceptance criteria

- [x] Add a distinct `--fine-detail-pursuit` option to `scripts/convert.py`/workflow plumbing;
      it is default off and mutually exclusive with the existing `--fine-detail`.
- [x] Run only after the ordinary safe schedule on masked normalized-renderer jobs.
- [x] Use fixed 128-row waves, sigma-1.5 residual high-pass scoring, 5x5 within-wave NMS,
      exact-site deduplication across waves, 0.35-pixel isotropic scale, opacity 0.8, and
      `margin + 6 px` eligibility.
- [x] Freeze every pre-tail row. After each wave, jointly solve only all accumulated pursuit-row
      colors with the exact normalized denominator and deterministic regularized CG.
- [x] Apply the full protected commit gate after every wave. Stop at the first safe state reaching
      `25%` sigma-1.5 high-pass and `20%` Laplacian reduction, on a rejected wave, insufficient
      sites, or 2,048 added rows.
- [x] Log baseline/current detail metrics, target progress, row/site counts and hashes, solve
      diagnostics, termination reason, time, and protected decisions in config/history/report
      outputs.
- [x] Focused tests cover default-off parity, option exclusivity, deterministic deduplication,
      partial-solve inheritance freeze, target stopping, rejection rollback, containment, and
      telemetry.
- [x] Update README, architecture, ADR, task index, ARA claim/evidence records, and verification
      documentation consistently; keep defaults unchanged.

## Depends on

FIT-005/031/033/038/039, CORE-010/011/012, BENCH-002, ADR-0029.

## Notes

The Janelle target values are mechanism defaults for this opt-in development path, not a
generality or optimality claim. Any default or efficiency promotion still requires independent
images, replicated seeds, and rate/work-matched controls.

## Result

The production replay stops at 768 rows and reproduces FIT-039's canonical site set, exact
non-color tensors, and colors within `7.16e-7`; every acceptance and protected check passes.
Deep sigma-1.5 high-pass/Laplacian MSE fall `25.926%/27.316%`, raw LPIPS falls `10.46%` relative,
and foreground PSNR gains `0.034326 dB`. FIT-041 supplies the valid same-base comparison to the
latest-commit tail. C59/C60 and ADR-0030 bind the default-off interface and narrow result.
