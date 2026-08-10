# FIT-028 / FIT-029 interior coverage budget — masked Janelle development screen

## Scope

Masked-arm half of FIT-028, plus the masked measurement FIT-029 required before any removal
decision. One exposed development image, three seeds, one GPU. There was no distinct prospective
reviewer, so this is a **development diagnostic**: it can kill a knob and inform pipeline design,
but it cannot promote a default. The full-frame Kodak-24 arm of FIT-028 and BENCH-018's granularity
question are untouched and still open.

## Protocol

Frozen in `tasks/FIT-028-interior-coverage-budget-screen.md` before the first fit, including the
reading rule and the guardrail below.

- **Source:** `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg`
  (SHA-256 `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`, native `5328x4608`)
  with `mask/mask_C0001.png` (SHA-256
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`). Fitted at `--max-side 1200`,
  i.e. the established `1200x1038` Janelle regime of FIT-023/C56/C60. Exposed development data in
  one capture group: not held out, not confirmation.
- **Arms:** the `hole_budget` stage registered in `workflows.STAGE_VARIANTS` — `current` (0.0),
  `budget1e4` (1e-4), `budget5e4` (5e-4), `budget2e3` (2e-3). The transform moves exactly one
  schedule field; a focused test asserts every phase budget is unchanged.
- **Budgets:** shipped recipe, capacity 11,000, `quadtree_wse`, `mask_margin` 0.75, exact CUDA
  renderer, seeds 0/1/2, RTX 3050.
- **Primary response:** terminal foreground PSNR. **Guardrails:** MS-SSIM, LPIPS, terminal
  `interior_hole_fraction`. **Declared reading rule:** recovering vetoed steps is not a result; a
  budget is interesting only if PSNR improves *and* terminal interior holes do not rise above
  `current`.

## Result: the frozen gate is not met

Means over three seeds; paired deltas use the seed as pairing key, with 95% Student-t intervals at
n=3 (`t=4.303`).

| arm | PSNR | ΔPSNR [95% CI] | ΔLPIPS | Δfit s | step acceptance | terminal interior holes |
|---|---:|---|---|---:|---:|---|
| `current` | 26.0329 ± 0.1806 | — | — | — | 8.71% | 0.00000% (3/3) |
| `budget1e4` | 26.1194 ± 0.1075 | +0.0865 [-0.1507, +0.3237] | -0.00028 | -83.5 | 9.26% | 0.00000% (3/3) |
| `budget5e4` | 25.9646 ± 0.0767 | -0.0683 [-0.5245, +0.3878] | -0.00036 | +93.3 | 9.53% | 0.00000% (3/3) |
| `budget2e3` | 26.2445 ± 0.1127 | +0.2116 [-0.5151, +0.9383] | -0.00084 | +77.8 | 10.48% | **0.00131%** |

No arm wins PSNR with an interval excluding zero, so **no nonzero budget is authorized**. The PSNR
response is also non-monotonic (`+0.086 / -0.068 / +0.212`), which is not a dose-response.

Two comparisons are nominally significant in isolation — `budget5e4` fit seconds
(`+93.3`, CI `[+36.7, +149.9]`) and `budget2e3` LPIPS (`-0.00084`, CI `[-0.00100, -0.00068]`).
Neither survives multiplicity: 3 arms x 5 responses is 15 comparisons, Bonferroni at n=3 requires
`|t| ~ 24.4`, and these reach `7.1` and `22.6`. They are reported and **not** treated as effects.

`budget2e3` is the arm the reading rule was written for. It has the largest PSNR point estimate, the
only nominal LPIPS gain, and the highest acceptance — and it is the only arm that retains interior
holes (`0.00131%` versus `0.00000%`). That breaches the pre-declared guardrail, so its gain is at
least partly bought by deleting coverage: the failure mode the task existed to distinguish, appearing
on the arm where it was predicted.

## Mechanism: the veto is a symptom, the tail guard is the cause

Block-level accounting over all three seeds. A rejected block records a list of reasons, so
`cited` counts blocks citing a reason and `alone` counts blocks where it is the *only* reason.
Relaxing one term cannot revive a block several terms rejected, so `alone` bounds what loosening
that term can recover.

| arm | rejected blocks | `interior_holes` cited | alone | `cvar99_mse` cited | alone |
|---|---:|---:|---:|---:|---:|
| `current` | 73 | 63 | 4 | 66 | 6 |
| `budget1e4` | 68 | 18 | 1 | 64 | 16 |
| `budget5e4` | 72 | 6 | 0 | 70 | 21 |
| `budget2e3` | 72 | **0** | 0 | 70 | **39** |

This is the result of the screen. Step acceptance rises monotonically with the budget
(8.71% -> 9.26% -> 9.53% -> 10.48%) and interior-hole citations collapse `63 -> 18 -> 6 -> 0`, so
the ADR-0026 mechanism works exactly as specified. **But the number of rejected blocks does not
move** (`73 -> 68 -> 72 -> 72`). At `budget2e3` the interior-hole veto is eliminated entirely and
discarded work is unchanged, because the rejections migrate to the CVaR99 tail guard, whose
sole-cause count rises `6 -> 39`.

Substitution is therefore complete on this arm. In the baseline, only 4 of 73 rejected blocks
(5.5%) were vetoed by the hole term alone, which bounded the recoverable set before any arm ran.
ADR-0026's premise — that the interior-hole veto is the *cause* of discarded work, worth 75% of it —
does not hold on the masked arm. The binding constraint is the CVaR99 tail guard together with the
boundary pixel-error terms.

## FIT-029: `safe_polish` is tolerance-starved, not veto-blocked

| arm | attempted | accepted |
|---|---:|---:|
| `current` | 1,404 | 0 |
| `budget1e4` | 1,404 | 0 |
| `budget5e4` | 1,872 | **31** |
| `budget2e3` | 1,404 | 0 |

FIT-029 required separating a vetoed phase from a miscalibrated one before any removal. The
separation is clean:

- Every `safe_polish` rejection in all 12 cells cites `boundary_mse_regressed` **and**
  `cvar99_mse_regressed` — the phase's own tightest-in-the-schedule pixel-error tolerances.
- `interior_holes_regressed` appears in `safe_polish` in only 2 of 12 cells, always co-cited, never
  alone. The shared interior veto is not what stops this phase here.
- `no_material_gain` fires in 3 cells, so `minimum_relative_gain` binds directly — FIT-029's
  hypothesis (2).
- `budget5e4` seed 1 accepted 31 of 936 attempted steps. This is the first nonzero `safe_polish`
  acceptance on record; prior evidence was 0 of 3,276 across 7/7 unmasked images.

So the phase is not structurally dead and **removal is not indicated**. The cause is tolerance-driven.
Retuning `minimum_relative_gain` and the polish pixel-error tolerances is a new question, not this
task's, and must not be tuned on this consumed frame.

## Integrity and limitations

- Bundle: `results/fit028_hole_budget_janelle_frame00008_2026-08-08/` (git-ignored);
  `manifest.json` SHA-256 `e46b26b0c6051de76b7ecb4cabb17d2234c9b534f0750d49b58a677bec5ad35a`.
  Manifest records commit `a8e8dde89a7560f4a13c356ebaf556a5d9b67437`, branch `main`,
  `dirty: true`, `status_sha256 ff69c26d…`.
- **`check_report_bundle.py` fails on the default gate** with 12 rows of
  `config_json repository identity differs from manifest`, and passes only under `--allow-dirty`.
  Cause: the working tree was edited *during* the grid, so per-cell provenance diverges from the
  end-of-run manifest. Every executed module was frozen for the whole window — `workflows.py` last
  modified `11:21:34`, before launch; `pipeline.py`/`safe_schedule.py`/`fit.py`/`init.py` untouched
  since July — while the edits inside the window (`README.md`, `docs/architecture.md`,
  `tasks/INDEX.md`, ADR-0026, the report driver, its tests) are files the run never imports. The
  science is not contaminated, but this is a real process defect: do not edit the tree while a grid
  executes.
- Recorded `source.path`/`mask_path` point at a session scratchpad, because `stage_search` needs a
  parallel mask tree with matching relative stems and the originals are `rgb/C0001.jpg` versus
  `mask/mask_C0001.png`. Those paths will not exist on replay; the content hashes above are the
  binding, and the staging step is in Reproduction.
- Arms do not land on equal realized counts (`current` 10,056/11,000/10,816 versus `budget2e3`
  11,000/10,972/11,000) even though `capacity` and `step_scale` are equal, so a count-driven gain
  must not be read as a budget effect. Every table reports N.
- One image, one capture group, exposed development data, three seeds, one RTX 3050, non-bit-
  reproducible CUDA. No held-out, default, generality, convergence-speed, or compression claim
  follows. `safe_polish` sees only 1-2 blocks per cell, so its rate is a consistent pattern over
  small samples, not a precise estimate.
- `blocks_citing_only` was added **after** the baseline arm was inspected. It is a read-only
  post-hoc mechanism diagnostic recomputed from persisted history and does not touch the frozen
  gate, which remains terminal PSNR plus the hole guardrail.

## Reproduction

```bash
# stage_search needs a parallel mask tree with matching stems
mkdir -p /tmp/janelle/images /tmp/janelle/masks
ln -s "$JANELLE/frame_00008/rgb/C0001.jpg"       /tmp/janelle/images/C0001.jpg
ln -s "$JANELLE/frame_00008/mask/mask_C0001.png" /tmp/janelle/masks/C0001.png

python scripts/stage_search.py /tmp/janelle/images \
  results/fit028_hole_budget_janelle_frame00008_2026-08-08 \
  --mask-dir /tmp/janelle/masks --stage hole_budget --seeds 0 1 2 --max-side 1200 --lpips

python scripts/experiments/fit028_bench018_gate_screen_report.py \
  results/fit028_hole_budget_janelle_frame00008_2026-08-08 --baseline current
```

The maintained `index.html` carries per-cell curves over attempted steps, native-resolution
target/reconstruction/error images, intermediate accepted states, and commit-gate accounting.
`comparison.html` adds the cross-arm view: paired deltas, per-phase acceptance, both rejection
footings with the sole-reason subset, and quality against wall-clock.
