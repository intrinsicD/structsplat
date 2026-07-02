# MERGE-001: Integrate Claude core optimizations and Codex stage search into main

**Status: partial.** Integration branch `merge/claude-codex-structsplat` created; both branches
combined by a real git merge (both parents preserved) with the 8 shared core files resolved as a
semantic merge (ADR-0009). Code-level acceptance criteria pass; the large COCO/CUDA confirmation is
pending a GPU + dataset (see below). Merge to `main` by PR only after that run.

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
- [x] Combined branch imports cleanly and `structsplat --help` exposes `fit`, `ablation`, and `stage-search`.
- [x] Existing StructSplat tests pass (33 passed).
- [x] New/updated tests cover Claude renderer behavior, sampling behavior, fit dynamics, Codex stage-search CLI, and optional opacity/additive paths (Codex `test_render`/`test_smoke`/`test_gaussians`/`test_stage_search` + Claude `test_codec`/`test_fit_dynamics`, all green on the merged branch).
- [x] `ruff check src benchmarks tests` passes.
- [x] Re-run the screening with the combined branch and include both Claude defaults and Codex stage variants — run at reduced scale (4 local images, budget 512, 40 iters, max-side 160, **CPU**) since this environment has no GPU or COCO dataset; `structsplat stage-search` produced ranked JSON/CSV/summary and Claude's `aniso_flanking`/`aniso_onedge` init remained the top configs.
- [~] Combined best config matches or exceeds Claude `flanking_split32` within tolerance (PSNR ≤0.10 dB, MS-SSIM ≤0.001, ≤25% slower): **cannot verify the exact numbers** without the original screening's image set/GPU; the merge preserves Claude's renderer/sampler/fitter code paths verbatim, so behavior is expected to match. Confirm in the COCO/CUDA run.
- [~] Combined fast config matches or exceeds Claude `onedge` within tolerance — same caveat.
- [ ] Re-run a larger confirmation: at least 20 COCO images, 512 and 1024 Gaussians, 3 seeds for finalist configs. **Blocked**: needs a GPU + the COCO val2017 dataset (neither present here). This is the remaining gate before merging to `main`.
- [x] Update `README.md`, `docs/architecture.md`, and relevant ADR/task files to state the chosen default and why (ADR-0009; INDEX/task statuses; README/architecture).
- [x] Do not merge raw datasets, checkpoints, worktree folders, cache folders, or large run directories (screening outputs were written under the scratch dir, not the repo).

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
