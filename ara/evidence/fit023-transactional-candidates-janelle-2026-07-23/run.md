# FIT-023 transactional candidates — Janelle C0001 development factorial

## Scope

Source-bound one-image, one-seed development mechanism test. It is not held-out evidence, a
repository-wide default promotion, a production convergence claim, an actual-rate result, or a
speedup claim.

## Protocol

- Source: `frame_00008/C0001`, RGB SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`, mask SHA-256
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`, seed
  `1559856117`.
- Clean source commit: `6e3cf0d1836bebeaa1c2a72227d238a53d876445`.
- Device: NVIDIA GeForce RTX 4090; Torch 2.7.0+cu126; CUDA runtime 12.6.
- Common pipeline: 4,500 quadtree-WSE + 500 explicit-boundary initialization, global
  refinement, `cuda_tiled`, 11,000-row capacity, identical phase/event budgets and full
  foreground/boundary/CVaR/hole/outside safe-commit gate.
- Factorial arms: control; state-matched checkpoints every 50 steps; post-topology event color
  solve; both mechanisms.
- Runs were sequential on one GPU. `--no-archive` excludes `.rtgsv` and all rate/codec claims.

Exact command:

```bash
PYTHONPATH=src /home/alex/miniconda3/bin/python \
  scripts/run_janelle_safe_schedule_factorial.py \
  --out runs/janelle_C0001_transactional_candidates_factorial_20260723
```

## Result

| arm | FG PSNR | boundary PSNR | CVaR99 | p99 | interior holes | boundary holes | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 26.566 | 10.878 | .172265 | .017764 | 2.544% | 31.690% | 382.5 s |
| checkpoint | **27.068** | **11.397** | .153313 | **.014318** | 1.436% | **28.491%** | 419.3 s |
| event color | 26.464 | 10.794 | .175957 | .018526 | 2.806% | 33.413% | 412.2 s |
| combined | 27.063 | 11.388 | **.152707** | .014527 | **1.424%** | 28.575% | 541.9 s |

Checkpoint-only improves every gated metric versus control and also improves the recorded p99
diagnostic: foreground/boundary MSE -10.92%/-11.27%, CVaR99/p99 -11.00%/-19.40%, and relative interior/boundary holes
-43.57%/-10.10%, at +9.61% total time. Four earlier field+Adam snapshots were actually committed.

Event color solve was selected in 20 event-only and 23 combined topology commits, yet event-only is
worse than control on every quality/coverage metric and 7.8% slower end to end. Combined passes the
predeclared control rule but, versus checkpoint-only, loses foreground, boundary, p99, and boundary
coverage and costs +29.26% total time; its CVaR99 and interior-hole gains are small. The bounded
decision is therefore checkpoint-only for Janelle development, with event color disabled.

## Scientist pass

- Four of four arms and all expected fields, configs, histories, native images, and HTML pages
  exist; the factorial source/config/environment equality check passes.
- Each final NPZ hash matches its run manifest.
- Cold reload and full recomputation of foreground/boundary MSE, CVaR99, p99, both hole fractions,
  and both outside metrics produced observed delta zero for all four arms in this environment.
  CUDA remains tolerance-reproducible rather than generally bit-exact.
- Independently reapplying the gate across accepted non-marker transitions finds zero failures;
  summed attempted/accepted steps match every top-level record.
- Native final reconstruction/error PNGs are RGB 3964x1444; all report/index links resolve.
- Exact outside render and raw coverage maxima are zero for every arm.
- The focused/relevant regression suite passed 189 tests and Ruff passed. The full repository-wide
  1,391-test suite was not executed.
- The cold-replay cache ELF hashes to
  `843b37b6046b7eeaae2fdc7c32086c6f4ab428e8428257b25f2eae123c8718d2`, but the runner did
  not archive/manifest the per-process ELF. Source snapshots and cold replay are bound; exact
  per-arm ELF identity is not claimed.
- Configured 0.1% interior and 1% boundary coverage targets remain unmet. Artifact
  `converged=true` means the polish transaction reached a deterministic fixed point, not total
  coverage.

## Artifacts

- Common report:
  `runs/janelle_C0001_transactional_candidates_factorial_20260723/report.md`
- Common comparison:
  `runs/janelle_C0001_transactional_candidates_factorial_20260723/index.html`
- Raw comparison:
  `runs/janelle_C0001_transactional_candidates_factorial_20260723/comparison.json`
- Machine-readable audit:
  `runs/janelle_C0001_transactional_candidates_factorial_20260723/audit.json`
- Per-arm resolved configs, histories, source snapshots, fields, checkpoints, and images:
  `runs/janelle_C0001_transactional_candidates_factorial_20260723/{control,pareto_checkpoint,event_color_solve,combined}/`

## Disposition

Confirm the checkpoint mechanism on this development image; refute event-color quality and
end-to-end speed promotion; narrow combined to a viable but inferior trade-off; leave defaults
unchanged. Multi-image, multi-seed frozen confirmation is required before broader promotion.
