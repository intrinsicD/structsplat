# Benchmarks

All benchmark scripts write machine-readable rows plus enough resolved configuration to make a run
self-describing. See the `benchmark` skill for the full protocol.

`ablation.py` runs the core experiment (`ABL-001`): `{init strategy} x {budget}` on fixed images,
scored on PSNR / MS-SSIM / LPIPS + iterations-to-target.

```
python -m benchmarks.ablation path/to/images --budgets 2000 5000 10000 20000 --iters 1500 --target-psnr 35
```

Outputs `ablation.json`, `ablation.csv`, `summary.md`. `fitness(rows, strategy, budget)` exposes the
scalar a co-scientist loop maximizes over init/sampling variants.

`stage_search.py` runs `ABL-002`: factorial or influence-mode sweeps across tensor, density,
sampling, orientation, color, scale-cap, renderer, loss, optimizer, refinement, and pyramid stages.

```
python -m benchmarks.stage_search path/to/images --mode influence --budgets 2048 --iters 500
```

`rate_distortion.py` evaluates the codec/QAT path (`COMP-001/003`) and records full codec/render
semantics per row.

`coco_fit_compare.py`, `cross_repo_matrix_compare.py`, `optimization_followup.py`, and
`quadtree_init_compare.py` are focused comparison/follow-up harnesses used by the ARA trace. They
are intentionally narrower than `stage_search.py`; use them when reproducing the specific evidence
entry that names them. Each accepts `--seeds` and reports aggregate mean/std over image x seed
rows; `--seed` remains as a single-seed compatibility alias.
