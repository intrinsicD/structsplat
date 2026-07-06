# MERGE-001: Integrate Claude core optimizations and Codex stage search into main

**Status: done.** Integration branch `merge/claude-codex-structsplat` was merged by PR #1
(`f49aa18`). Both source branches were combined by a real git merge (both parents preserved) with
the 8 shared core files resolved as a semantic merge (ADR-0009). The remaining COCO/CUDA
confirmation gate completed on 2026-07-06; evidence:
`ara/evidence/merge001-coco-cuda-confirmation-2026-07-06/`.

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
- [x] Combined best/fast configs confirmed on a reproducible replacement protocol. The exact
      historical Claude `flanking_split32`/`onedge` numbers remain non-comparable because the
      original image set/protocol is not reconstructable from this repo, but the large COCO/CUDA
      gate now records the merged rows directly.
- [x] Re-run a larger confirmation: 20 COCO val2017 images, 512 and 1024 Gaussians, 3 seeds for
      finalist configs. Completed 720/720 exact-CUDA cells with zero errors on 2026-07-06.
- [x] Update `README.md`, `docs/architecture.md`, and relevant ADR/task files to state the chosen default and why (ADR-0009; INDEX/task statuses; README/architecture).
- [x] Do not merge raw datasets, checkpoints, worktree folders, cache folders, or large run directories (screening outputs were written under the scratch dir, not the repo).

## Completion notes

- 2026-07-06: Ran the larger MERGE-001 confirmation with exact CUDA on the first 20 sorted COCO
  val2017 images, budgets {512, 1024}, seeds {0,1,2}, 40 iterations, max-side 160, and six
  merged/finalist configs. The run completed 720/720 cells with zero errors.
- Overall mean PSNR ranked `codex_stage_top1` first (27.3443 dB), followed by
  `merged_onedge_fast` (27.2016 dB), `merged_shipped_flanking` (27.0827 dB),
  `merged_best_exact_cuda` (26.7006 dB), `codex_stage_top2` (26.6157 dB), and
  `codex_stage_top3` (26.4007 dB).
- Paired against `merged_shipped_flanking`, `codex_stage_top1` gained +0.2616 dB PSNR
  (108/120 wins), and `merged_onedge_fast` gained +0.1189 dB PSNR (100/120 wins).
- The older `merged_best_exact_cuda` feature-cap/residual-tensor row lost -0.3822 dB paired PSNR
  versus shipped flanking, so it should stay an experimental stage-search row rather than become a
  promoted default from MERGE-001.

## Validation commands
```bash
PYTHONPATH=src:. pytest -q tests
PYTHONPATH=src:. ruff check src benchmarks tests

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. python - <<'PY'
# Calls benchmarks.stage_search.run_stage_search once per finalist config.
# See ara/evidence/merge001-coco-cuda-confirmation-2026-07-06/run.md.
PY
```

## Non-goals
- Do not merge the parent OmniLatent training changes as part of this StructSplat task.
- Do not pick a final publication-quality default from the 4-image screening alone.
- Historical process note: do not push directly to `main`; future follow-up fixes should still merge by PR.

## Depends on
CORE-001/002, INIT-003/004, FIT-001, HIER-001, BENCH-001, ABL-002, COMP-001.
