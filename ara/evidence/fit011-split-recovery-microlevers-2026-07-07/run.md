# FIT-011 split-recovery micro-levers

Date: 2026-07-07

## Split-recovery smoke

Command:

```bash
PYTHONPATH=. python -m benchmarks.stage_search \
  results/datasets/abl004/kodak24/kodim01.png \
  results/datasets/abl004/kodak24/kodim07.png \
  --mode factorial --budgets 80 --seeds 0 --iters 60 --max-side 64 \
  --strategies aniso_flanking --tensor-operators central --tensor-colors luma \
  --density-modes structure --sampling-modes density_random --orientation-modes tensor \
  --color-modes bilinear --scale-modes spacing --scale-cap-modes none \
  --opacity-modes none --renderers normalized --aa-dilations 0.0 \
  --color-basis-modes constant --color-solve-modes none \
  --pixel-losses l1 --optimizers adam --lr-schedules none \
  --refine-sites residual --refine-primitives duplicate moment_preserving --refine-nms-modes off \
  --refine-color-inits target --refine-prune-modes off --refine-relocate-modes off \
  --state-seed-modes off on --row-temper-modes off warmup5 --support-fade-modes off \
  --pyramid-modes single --split-every 30 --split-count 16 --log-every 1 \
  --chunk 64 --outdir results/fit011_split_recovery_smoke_2026_07_07 \
  --device cpu
```

Artifacts: `split_recovery/config.json`, `split_recovery/stage_search.json`,
`split_recovery/stage_search.csv`, `split_recovery/summary.md`, `split_recovery/index.html`.

| primitive | state seed | row temper | mean PSNR | mean AUC | post-split delta | recovery iters | fit s |
|---|---|---|---:|---:|---:|---:|---:|
| duplicate | off | off | 23.6248 | 21.8625 | +0.1747 | 1.00 | 0.3193 |
| duplicate | off | warmup5 | 23.5868 | 21.8475 | +0.1747 | 1.00 | 0.3177 |
| duplicate | on | off | 23.4191 | 21.8203 | +0.1747 | 1.00 | 0.3254 |
| duplicate | on | warmup5 | 23.3882 | 21.8128 | +0.1746 | 1.00 | 0.3159 |
| moment_preserving | off | off | 23.3584 | 21.7844 | -0.0014 | 1.50 | 0.3185 |
| moment_preserving | off | warmup5 | 23.3490 | 21.7743 | -0.0014 | 1.50 | 0.3247 |
| moment_preserving | on | off | 23.2439 | 21.7589 | -0.0015 | 1.50 | 0.3171 |
| moment_preserving | on | warmup5 | 23.2323 | 21.7528 | -0.0014 | 1.50 | 0.3180 |

Decision: no split-recovery lever is promoted. State seeding and warmup did not improve the
measured post-split delta or recovery lag, and they reduced PSNR/AUC in this smoke. Keep both as
explicit stage-search controls only.

## Scheduled support-fade smoke

Cheap command:

```bash
PYTHONPATH=. python -m benchmarks.stage_search \
  results/datasets/abl004/kodak24/kodim01.png \
  results/datasets/abl004/kodak24/kodim07.png \
  --mode factorial --budgets 80 --seeds 0 --iters 60 --max-side 64 \
  --strategies aniso_flanking --tensor-operators central --tensor-colors luma \
  --density-modes structure --sampling-modes density_random --orientation-modes tensor \
  --color-modes bilinear --scale-modes spacing --scale-cap-modes none \
  --opacity-modes none --renderers normalized --aa-dilations 0.0 \
  --color-basis-modes constant --color-solve-modes none \
  --pixel-losses l1 --optimizers adam --lr-schedules none \
  --refine-sites none --refine-primitives duplicate --refine-nms-modes off \
  --refine-color-inits target --refine-prune-modes off --refine-relocate-modes off \
  --state-seed-modes off --row-temper-modes off --support-fade-modes off on until0.5 \
  --pyramid-modes single --log-every 1 \
  --chunk 64 --outdir results/fit011_support_fade_schedule_smoke_2026_07_07 \
  --device cpu
```

Artifacts: `support_fade/config.json`, `support_fade/stage_search.json`,
`support_fade/stage_search.csv`, `support_fade/summary.md`, `support_fade/index.html`.

| support fade | mean PSNR | mean AUC | fit s | mean fade alpha |
|---|---:|---:|---:|---:|
| off | 23.8361 | 22.1932 | 0.2996 | 0.0000 |
| on | 23.8668 | 22.1907 | 0.3079 | 1.0000 |
| until0.5 | 23.8530 | 22.1814 | 0.2997 | 0.5917 |

Decision: scheduled fade is not promoted. The schedule did not meet the intended criterion of
AUC >= fade-on while preserving fade-off final PSNR. Keep `support_fade=until<F>` searchable for
larger support-fade protocol runs, but default remains fade off.

## Scheduled support-fade 5k/10k budget smoke

Command:

```bash
PYTHONPATH=. python -m benchmarks.stage_search \
  results/datasets/abl004/kodak24/kodim01.png \
  results/datasets/abl004/kodak24/kodim07.png \
  --mode factorial --budgets 5000 10000 --seeds 0 --iters 60 --max-side 64 \
  --strategies aniso_flanking --tensor-operators central --tensor-colors luma \
  --density-modes structure --sampling-modes density_random --orientation-modes tensor \
  --color-modes bilinear --scale-modes spacing --scale-cap-modes none \
  --opacity-modes none --renderers normalized --aa-dilations 0.0 \
  --color-basis-modes constant --color-solve-modes none \
  --pixel-losses l1 --optimizers adam --lr-schedules none \
  --refine-sites none --refine-primitives duplicate --refine-nms-modes off \
  --refine-color-inits target --refine-prune-modes off --refine-relocate-modes off \
  --state-seed-modes off --row-temper-modes off --support-fade-modes off on until0.5 \
  --pyramid-modes single --log-every 5 \
  --chunk 64 --outdir results/fit011_support_fade_schedule_budget_smoke_2026_07_07 \
  --device cpu
```

Artifacts: `support_fade_budget/config.json`, `support_fade_budget/stage_search.json`,
`support_fade_budget/stage_search.csv`, `support_fade_budget/summary.md`,
`support_fade_budget/index.html`.

| budget | support fade | mean PSNR | mean AUC | fit s | mean fade alpha |
|---:|---|---:|---:|---:|---:|
| 5000 | off | 47.8790 | 39.4450 | 0.6017 | 0.0000 |
| 5000 | on | 47.9707 | 39.7002 | 0.6179 | 1.0000 |
| 5000 | until0.5 | 47.9724 | 39.6297 | 0.6256 | 0.5769 |
| 10000 | off | 49.1020 | 40.4712 | 0.9792 | 0.0000 |
| 10000 | on | 49.3857 | 40.8143 | 1.0200 | 1.0000 |
| 10000 | until0.5 | 49.3396 | 40.7349 | 1.0051 | 0.5769 |

Decision from the budget smoke: scheduled fade preserves or beats fade-off final PSNR, but AUC is
below always-on fade at both 5k and 10k. That fails the promotion rule, so the schedule stays
opt-in/searchable.
