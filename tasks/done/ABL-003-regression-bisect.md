# ABL-003: Bisect the undiagnosed −0.794 dB flagship regression

**Status: done.** From the 2026-07-03 repo review. **Investigation task — gates trust in all
post-merge tuning conclusions.**

## Context
The research trace records (ara/trace/exploration_tree.yaml, N03 → N04) that the flagship
`aniso_flanking` config dropped from **24.833 dB to 24.039 dB (−0.794)** on the same
four-image COCO benchmark (max-side 160, 512 Gaussians, 80 iters, seed 0) immediately after
"the update" (the PR #2 merge window), while random-init baselines moved < 0.01 dB and
Image-GS's analogue moved −0.455 dB (suggesting a shared-code change, e.g. renderer/fitter,
not an init-only change). The regression was recorded but never diagnosed. Every subsequent
"improvement" (N06: 24.29) is still below the pre-update number, so all post-merge tuning
conclusions float on an unexplained baseline shift.

## Goal
Name the commit and mechanism behind the −0.794 dB shift; decide whether it was a real quality
regression (fix it) or an intentional semantic change (document it and re-baseline the trace).

## Result
Implemented `benchmarks/regression_bisect.py` and ran:

```bash
python -m benchmarks.regression_bisect --download --device cpu
```

Evidence is committed under `ara/evidence/abl003-regression-bisect-2026-07-03/`.

| Commit | Mean PSNR | Delta vs previous | Verdict |
|---|---:|---:|---|
| `f49aa18` | 24.8202 | - | PR #2 merge base |
| `71fad3e` | 24.8202 | 0.0000 | stage-search code did not move this baseline |
| `ef730a9` | 24.0510 | -0.7692 | offending semantic/correctness change |
| `a455e98` | 24.0510 | 0.0000 | not the cause for default `color_mode=bilinear` |

Verdict: the N03->N04 StructSplat drop is an intentional semantic re-baseline, not an unfixed
quality regression. The shift localizes to `ef730a9`, whose diff changes blue-noise placement
spacing from exclusion radius to cell-side spacing (`sqrt(pi)` larger) and also includes clipped
renderer support, final-iteration restructure hygiene, and opacity padding fixes. N03/N04 evidence
and trace text are annotated accordingly.

## Acceptance criteria
- [x] Re-run the four-image benchmark at each of PR #2's commits (a455e98, ef730a9, 71fad3e)
      plus the merge base; per-commit PSNR table recorded.
- [x] The offending change identified at the diff level and its mechanism explained (candidate
      suspects worth checking first: the ef730a9 "correctness and experimental-validity fixes
      across renderer, fit, pyramid, codec" — a correctness fix can legitimately lower
      measured PSNR; and the a455e98 two_sided change).
- [x] Verdict recorded as a new trace node: either "regression, fixed in <commit>" (with the
      recovered number) or "intentional semantic change, N03 and N04 are not comparators"
      (with N03/before-update evidence entries annotated accordingly).
- [x] The four-image benchmark command + environment logged into `ara/evidence/` so the
      bisect itself is reproducible.

## Interfaces touched
`ara/trace/exploration_tree.yaml`, `ara/staging/observations.yaml`, `ara/evidence/`;
potentially a fix commit in `src/` depending on the verdict.

## Depends on
— (needs only git history + the existing coco_fit_compare/cross_repo harness; portable-path
fix from BENCH-002 helps but a local dataset path override suffices).
