---
name: structsplat-experiment
description: Plan, execute, validate, and report reproducible StructSplat research experiments. Use for comparative questions such as "does X help?", ablations, sweeps, hypothesis or killing tests, formal evidence runs, experiment report/index.html requirements, learning-curve requests, and logging positive or negative outcomes. Do not use for an ordinary implementation-only smoke test with no comparative or evidentiary question.
---

# StructSplat experiment workflow

Use the repository's existing authorities to turn a research question into a reproducible result.
This skill coordinates the lifecycle; it does not create another experiment registry.

## Load the supporting skills

1. Load `structsplat-core` first for repository invariants and current-state routing.
2. Use `structsplat-task-workflow` to open, hand off, review, and close substantial work.
3. Load `structsplat-benchmark` for maintained harness commands, comparability, metrics, and
   benchmark-specific caveats.
4. After execution, use `structsplat-results-audit` for the adversarial semantic pass.
5. Use `structsplat-review` and `structsplat-docs-sync` before final handoff.

`tasks/INDEX.md` plus the owning task are the sole task and frozen-protocol authority. `ara/` is
the sole claim and evidence authority. Do not add `docs/EXPERIMENTS.md`, a generic experiment
contract registry, or a parallel current-state tree.

## Classify the run before spending compute

### Diagnostic or smoke run

Use a small, clearly disposable run only to check wiring, feasibility, memory, or artifact shape.
It may use a dirty tree or reduced resolution/budget only when those qualifications are explicit.
Never use it to promote a method, close a scientific question, change a default, or consume a
sealed confirmation split. Label its output diagnostic.

### Formal result-bearing run

Use this path for any comparison, quantitative conclusion, default decision, public claim, or task
closure that depends on outcomes. A formal run requires all of the following before execution:

- an owning task with a frozen protocol;
- a distinct prospective reviewer approving the exact protocol digest with
  `Outcomes accessed: No`;
- a clean, identified source commit and resolved environment;
- a new immutable output directory; and
- explicit data roles, budgets, seeds, metrics, controls, and killing/stop rules.

Do not repair, overwrite, or selectively rerun cells inside an executed formal bundle. Correct the
driver or protocol, create a new output directory, and retain the invalid or superseded run.

## Freeze the protocol in the task

Record enough information that another agent can execute the run without making scientific
choices:

- question, hypothesis, null, and the decision the result is allowed to support;
- source images/dataset, hashes or immutable identifiers, inclusion rules, and train/development,
  validation, confirmation, or held-out roles;
- candidate arms and strongest relevant controls, distinguishing official native methods from
  local mechanism transplants;
- renderer, fitter, initializer, checkpoint rule, clamp/scoring convention, and all resolved
  configuration axes;
- equal iteration, Gaussian-count, byte/rate, search, wall-clock, and hardware budgets as relevant;
- seeds, repeat count, pairing key, aggregation unit, uncertainty method, and missing/error policy;
- primary metrics, guardrails, minimum effect, stop rule, killing gate, and forbidden follow-ups;
- exact command, output directory convention, source commit/dirty policy, and environment identity;
- expected raw tables, field/image artifacts, temporal telemetry, and report contents.

When a split exists, tune and select only on development/training plus a frozen validation slice.
Held-out or confirmation data is reporting-only and must not rescue a failed development gate. If
a blinded test trajectory is part of the question, preregister it, produce it without checkpoint
selection, and reveal it only after all choices are frozen.

Append the prospective `### Protocol review` block from `tasks/README.md`. The declared digest
scope and its recomputation command are task-specific; the generic task checker validates the
review block but does not infer the protocol bytes.

## Choose the execution surface

Prefer the maintained current-profile workflows:

```bash
python scripts/benchmark.py SOURCE OUTDIR [flags]
python scripts/ablation.py SOURCE OUTDIR [flags]
python scripts/stage_search.py IMAGE OUTDIR --stage STAGE [flags]
python scripts/convert.py SOURCE OUTDIR [flags]
```

Use `convert.py` for a single current-profile artifact or calibration, `benchmark.py` for the
current profile plus optional native baselines, `ablation.py` for the fixed stage-removal matrix,
and `stage_search.py` for registered variants of one stage on one image.

If none can express the frozen question, add a bounded driver under
`scripts/experiments/<TASK-ID>_<slug>.py`. Keep reusable implementation in `src/structsplat/` or
`benchmarks/`; the driver should only bind the protocol and serialize results. Document the exact
reproduction command in its module docstring and task. Do not add another top-level launcher.

Before a formal run, record at least:

```bash
git status --short
git rev-parse HEAD
```

The formal tree must be clean. CUDA atomic accumulation is not bit-reproducible; bind GPU model,
renderer, package/CUDA versions, seed, and source rather than claiming bit-exact GPU replay.

## Preserve comparison validity

- Hold inputs/splits, renderer equation, fitter, horizon, checkpoint rule, scoring convention,
  and candidate/search budget fixed across arms unless the difference is the preregistered axis.
- Enforce the same final Gaussian or actual-byte budget when making capacity or compression
  comparisons. Record `n_gaussians`; for actual-rate claims measure the complete persisted stream.
- Preserve seed/image pairing and use multiple seeds when claiming a stable difference.
- Keep failed, missing, stopped, and non-finite cells visible. Never summarize only survivors.
- Record attempted steps and nominal horizon. An early-exit curve is not an exact terminal-quality
  comparison unless the protocol explicitly defines that estimand.
- Do not call a local reimplementation an official native baseline. Record upstream revision,
  environment, native output, and central rescoring for native comparisons.

The detailed ABL/BENCH rules remain authoritative in `structsplat-benchmark` and the owning task;
do not copy or silently loosen them here.

## Require a portable result bundle

Maintained workflows write these root artifacts:

- `manifest.json` with command, profile, source inputs, variants, seeds, and repository identity;
- `metrics.json`, `metrics.jsonl`, and `metrics.csv` with the same stable result rows;
- `index.html`, using only portable relative links.

Each successful current-profile cell also links its resolved `config.json`, `history.json`, fitted
`field.npz`, target image, full-resolution reconstruction, absolute-error visualization, and any
intermediate reconstruction/error snapshots. Native-baseline cards and their own report links are
included only when those baselines were requested and completed.

Run the structural bundle gate before interpretation or handoff:

```bash
python scripts/check_report_bundle.py OUTDIR
```

`--allow-dirty` and `--allow-error-cells` permit diagnostic inspection only. They do not turn the
bundle into claim-ready evidence.

## Current `index.html` contract

The maintained static 2D report contains:

- the report title, current evidence scope, executed command, and links to the manifest and all
  three raw-metric formats;
- a run matrix with method, image, seed, Gaussian count, final PSNR, MS-SSIM, optional LPIPS,
  fit seconds, and total seconds;
- an explicit error section for failed cells;
- for each current-profile run, its identity, final metrics, phase timings, optional
  error-tail/pursuit summaries, and links to `field.npz`, `history.json`, and `config.json`;
- clickable target, reconstruction, and absolute-error images whose linked PNGs retain their
  native resolution, plus intermediate reconstruction/error checkpoints;
- current-profile curves over **attempted steps** for PSNR, SSIM, MS-SSIM, LPIPS, MSE, MAE,
  CVaR99 MSE, p99 MSE, interior-hole fraction, boundary-hole fraction, and cumulative elapsed
  seconds; and
- optional official-native-baseline cards and links to their standalone reports.

For the maintained single-image fitting workflows, these curves are reconstruction/optimization
metrics against the fitted target. They are not separate train and test errors. The page is a
portable 2D report, not an interactive 3D/RTGSV viewer. If a new experiment requires separate
training, validation, test, Gaussian-count, VRAM, or topology trajectories, add and test that
telemetry and report surface **before** the formal run, then freeze it in the task. Do not infer
missing curves from terminal metrics after execution.

For visual handoff, open `OUTDIR/index.html` in a real browser, click the linked PNGs to inspect
native pixels, and verify representative raw-artifact links. The bundle checker establishes
structural portability; browser inspection catches presentation and full-resolution handoff
failures.

## Audit and record the outcome

After the structural gate:

1. Run `structsplat-results-audit` as an adversarial referee pass.
2. Recompute primary comparisons and gate predicates from raw rows, including missing/error cells.
3. Reconcile source/config/split identity, actual resource scope, native/local labels, and figures.
4. Classify the result as diagnostic, unavailable, negative/refuted, inconclusive, or supported at
   the exact measured scope.
5. Preserve negative and failed outcomes; do not retune a frozen gate or silently replace rows.
6. Update the task decision and handoff. Before any quantitative/capability statement enters
   `README.md`, `docs/`, an ADR, or task status, bind it to an ARA observation/claim and tracked
   evidence according to `CLAUDE.md`.

A structurally valid bundle is not semantic approval. Prospective protocol review is not outcome
review. Self-review is provisional, never independent acceptance.

## Completion checklist

- The task and Index agree; `tasks/SESSION-BRIEF.md` is regenerated.
- Focused driver/report tests pass.
- Every formal bundle passes `scripts/check_report_bundle.py` without diagnostic allowances.
- The browser report exposes the promised curves, native-resolution image links, and raw artifacts.
- The results audit and ARA disposition match the actual evidence scope.
- `structsplat-review`, `structsplat-docs-sync`, and `./scripts/verify.sh` pass before handoff.
