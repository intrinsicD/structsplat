# MERGE-001: Integrate Claude core optimizations and Codex stage search into main

**Status: todo.** Create an integration branch from `main`, combine both experimental branches, and merge only after the combined result beats the current branch-local screenings.

## Source branches
- Claude optimized core: `origin/claude/approach-review-optimize-gkhhds` (`f465f57`).
- Codex stage-search harness: `origin/codex/stage-search-ablation-20260702` (`a70b153`).
- Supporting parent-repo comparison artifacts: `origin/codex/structsplat-stage-search-results-20260702` (`368e493`).

## Goal
Ship one StructSplat `main` that keeps Claude's stronger renderer/sampling/fitter behavior and Codex's broader ablation/stage-search infrastructure.

The current screening evidence is:
- Claude `flanking_split32`: mean PSNR 28.6573, MS-SSIM 0.97879, 0.287 sec/image.
- Claude `onedge`: mean PSNR 28.5279, MS-SSIM 0.97877, 0.252 sec/image.
- Codex best quality: mean PSNR 25.3951, MS-SSIM 0.95656, 5.406 sec/image.
- Codex fast quality: mean PSNR 24.8218, MS-SSIM 0.95270, 0.166 sec/image.

## Integration plan
1. Start from current `main`:
   `git switch main && git pull && git switch -c merge/claude-codex-structsplat`.
2. Port Claude's core changes first:
   - Ragged/tight-support normalized renderer and activity computation.
   - Faster WSE/anisotropic sampling implementation.
   - Retuned fitter learning rates.
   - Adam state carry across prune/split.
   - `aa_dilation` support.
   - Codec and rate-distortion files if they remain cleanly separable.
3. Port Codex's stage-search layer on top:
   - `structsplat stage-search` CLI.
   - `benchmarks/stage_search.py`.
   - Stage alternatives: tensor operator, density mode, sampling mode, color mode, scale mode, opacity mode, renderer mode, loss, optimizer, LR schedule, refinement, pyramid.
   - Screening script and `ABL-002` task.
4. Reconcile config defaults:
   - Keep Claude's retuned LR defaults as production defaults.
   - Keep Codex's extra config fields as optional knobs.
   - Ensure stage-search can reproduce Claude defaults as named candidates.
5. Remove or demote weak options:
   - Additive renderer must remain non-default and marked experimental unless fixed.
   - Pyramid and residual-add should remain candidates, not defaults, until a larger run proves them.

## Acceptance criteria
- [ ] Combined branch imports cleanly and `structsplat --help` exposes `fit`, `ablation`, and `stage-search`.
- [ ] Existing StructSplat tests pass.
- [ ] New/updated tests cover Claude renderer behavior, sampling behavior, fit dynamics, Codex stage-search CLI, and optional opacity/additive paths.
- [ ] `ruff check src benchmarks tests` passes.
- [ ] Re-run the 4-image screening with the combined branch and include both Claude defaults and Codex stage variants.
- [ ] Combined best config matches or exceeds Claude `flanking_split32` within tolerance: PSNR no worse than 0.10 dB, MS-SSIM no worse than 0.001, and total time no more than 25% slower.
- [ ] Combined fast config matches or exceeds Claude `onedge` within tolerance or clearly documents the speed/quality tradeoff.
- [ ] Re-run a larger confirmation: at least 20 COCO images, 512 and 1024 Gaussians, 3 seeds for finalist configs.
- [ ] Update `README.md`, `docs/architecture.md`, and relevant ADR/task files to state the chosen default and why.
- [ ] Do not merge raw datasets, checkpoints, worktree folders, cache folders, or large run directories.

## Validation commands
```bash
PYTHONPATH=src:. pytest -q tests
PYTHONPATH=src:. ruff check src benchmarks tests

PYTHONPATH=src:. python -m structsplat.cli stage-search \
  ../../data/coco2017/val2017/000000186042.jpg \
  ../../data/coco2017/val2017/000000190140.jpg \
  ../../data/coco2017/val2017/000000444879.jpg \
  ../../data/coco2017/val2017/000000554838.jpg \
  --budgets 512 \
  --iters 40 \
  --max-side 160 \
  --device cuda \
  --outdir results/merge_001_screening
```

## Non-goals
- Do not merge the parent OmniLatent training changes as part of this StructSplat task.
- Do not pick a final publication-quality default from the 4-image screening alone.
- Do not push directly to `main`; merge by PR after the acceptance criteria pass.

## Depends on
CORE-001/002, INIT-003/004, FIT-001, HIER-001, BENCH-001, ABL-002, COMP-001.
