# FIT-040/041 orthogonal fine-detail pursuit — Janelle full-frame development screen

## Outcome

The new default-off orthogonal pursuit is the clear winner for the predeclared deep fine-detail
objective on the requested masked Janelle `frame_00008/C0001` image. Starting from the persisted
11,000-row current-pipeline field, it reaches the first protected-safe target after six 128-row
waves (768 rows): deep sigma-1.5 high-pass MSE falls `25.9262%`, Laplacian MSE falls `27.3157%`,
and raw LPIPS falls `10.4601%` relative. Every inherited row remains bit-for-bit frozen, outside
metrics stay exactly zero, and the full protected gate passes.

The user's latest-commit FIT-031 idea did work, but it solved a different problem. On the exact
same base, its `0.5` effective-support tail spends all 2,777 requested rows and gains
`0.320788 dB` foreground PSNR, versus pursuit's `0.034326 dB`. However, all 2,777 final row
centers lie within 2.24 px of the mask boundary and zero lie in the predeclared deep region.
Deep high-pass changes only `0.000042%` relative, Laplacian changes `0%`, and raw LPIPS improves
`3.2114%`. It is a useful boundary/global-error repair on this frame, not an effective fabric
fine-detail allocator.

## Scope correction

FIT-031's committed evidence used a `1200x437` subject crop. FIT-039/040 use the requested full
`1200x1038` masked frame. Therefore the earlier raw `4,608` versus `768` row counts are not a
valid efficiency comparison. FIT-041 was added before claim crystallization to run the unchanged
FIT-031 tail from FIT-040's exact base field, decoded target pixels, mask, renderer, seed, and
protected baseline. All five binding checks pass.

This is one exposed image, one seed, one RTX 3050, and two unequal-work terminal policies. It is
not an equal-rate, equal-work, held-out, default, generality, codec, or state-of-the-art result.

## Same-base result

| terminal tail | added rows | deep rows | sigma-1.5 HP reduction | Laplacian reduction | raw LPIPS reduction | FG PSNR gain |
|---|---:|---:|---:|---:|---:|---:|
| FIT-031 MAE effective support | 2,777 | 0 | 0.000042% | 0.0000% | 3.2114% | **+0.320788 dB** |
| orthogonal detail pursuit | **768** | **768** | **25.9262%** | **27.3157%** | **10.4601%** | +0.034326 dB |

The error tail uses `3.6159x` as many rows. Its final-center SDF quantiles
`p0/p10/p50/p90/p100` are `2/2/2/2.236/2.236 px`; pursuit's are
`7/9/24.718/45.591/67.231 px`. The result is therefore an objective trade-off with a directly
observed allocation mechanism, not merely two endpoint scores.

## Selected mechanism

After ordinary `safe_polish`, `--fine-detail-pursuit`:

1. measures the current RGB sigma-1.5 high-pass rendering residual only at mask pixels deeper than
   `margin + 6 px`;
2. selects 128 maxima with 5x5 within-wave NMS, forbidding only exact prior sites across waves;
3. appends ordinary 0.35-pixel isotropic, opacity-0.8 constant-color Gaussians;
4. jointly solves every accumulated pursuit-row color with deterministic regularized CG under the
   exact normalized-compositor denominator, while every inherited tensor stays frozen;
5. applies the unchanged protected gate and remeasures; and
6. stops at the first safe `25%` high-pass plus `20%` Laplacian state, rejection, site exhaustion,
   or 2,048 added rows.

FIT-032--036 killed gauge-lifted dipoles, a spectral color objective, sparse affine appearance,
and residual-ridge anisotropy as explanations for the missing effect. FIT-037 showed one-shot
ranking saturates at `15.01%/12.04%` with 2,048 rows. FIT-038 improved that to
`20.22%/16.21%` by iterative remeasurement but its 5x5 cross-wave exclusion was too strict.
FIT-039's killing ablation retained within-wave NMS and relaxed cross-wave exclusion to exact-site
deduplication; this is the first configuration to cross the frozen target, at 768 rows.

## Scientist pass

- The production FIT-040 replay reaches the same 768-site canonical set as FIT-039. After
  canonical row ordering, all non-color tensors are exact and colors differ by at most
  `7.1526e-7`, consistent with CUDA reduction order. Ordered and canonical set hashes are both
  retained.
- FIT-039's cold field audit exactly reproduces every stored metric, confirms 768 unique sites,
  exact-zero outside values, no-op containment projection, the protected gate, and a safe
  color-solve fixed point.
- All six production waves are protected-safe, all inherited prefixes are exact, waves one
  through five miss the target, and wave six is the first pass.
- The same-base FIT-031 control accepts all 2,777 requested rows in six waves, then rejects its
  first global convergence block and stops at a deterministic fixed point. Its protected metrics
  improve and containment remains exact.
- Perceptual scoring uses the repository BENCH-001 metric implementation on the same matted target.
  Pursuit improves raw LPIPS `0.01019005 -> 0.00912416`; the error tail improves it to
  `0.00986280`.
- The pursuit effect is spatially concentrated: `11.33%` of deep pixels improve, `8.88%` worsen,
  and `42.27%` of qualifying 32-pixel tiles have positive reduction. Net energy falls strongly,
  but this is not uniform-detail evidence.
- The independent JSON audit has 17/17 checks true and explicitly records pursuit as the
  fine-detail winner, the error tail as the global foreground-PSNR winner, and all promotion flags
  false.

## Reproduction

The persisted base is already post-schedule. Both replay drivers make the otherwise repeated entry
color solve an exact float32 no-op with one iteration and finite `1e30` ridge; each tail then uses
its own unchanged solver/configuration.

```bash
python3 scripts/experiments/fit040_janelle_production_pursuit.py --quiet
python3 scripts/experiments/fit041_janelle_equal_base_error_tail.py --quiet
python3 scripts/experiments/evaluate_fit039_perceptual.py \
  --result runs/fit039_janelle_exclusion_screen_20260728/result.json \
  --field runs/fit041_janelle_equal_base_error_tail_20260728/field.npz \
  --out runs/fit041_janelle_equal_base_error_tail_20260728/perceptual.json \
  --device cuda:0
python3 scripts/experiments/audit_fit041_equal_base_tail_comparison.py
```

## Durable files

- `production_result.json` — SHA-256
  `b308294482adeaaea11756e1e74eca2261258012c2e0ea523b3e6601bba8e60b`.
- `equal_base_error_tail_result.json` — SHA-256
  `1abd8ac1d617756b1a5a14cc367d9acb99525e7a7e5c4034dedaf781b64657fa`.
- `equal_base_error_tail_history.json` — SHA-256
  `3a131d002491f653cb3ff39a9e76ed727c12b477d03649038a1acf065a3d0860`.
- `audit.json` — SHA-256
  `bed519d358d8a5f61cba9ad22c9e84a4c76350340acddde796eba625bfb18c9f`.
- `prototype_audit.json` — SHA-256
  `fa2869ec0f2eee7ea5eff4da3f38db31639bbc6dd8c5326e2f770555c5e21749`.
- `pursuit_perceptual.json` / `error_tail_perceptual.json` — SHA-256
  `7b131c000e14c901a1ad9487ca58f711fdb1e3100c06183aa2d6ab48b9dc9139` /
  `8a16b8b69a7efe98dd5840e3c79940ac0643ad30cd2282f2bc6964b8c13fd139`.
- `spatial.json` — SHA-256
  `4b0dfb0d2b95fe41794fbbbda7f97cc9c25531f2c0f6e6796f867d4ef55e9a2f`.

Focused pursuit/pipeline verification passed `75` tests with `4` slow tests deselected. After
documentation and ARA synchronization, the final repository-wide `./scripts/verify.sh` passed:
ruff clean; `1,488` tests passed, `4` skipped, and `514` deselected; `docs_sync`, `check_ara`
(`60` claims), `check_task_policy`, and `check_script_layout` all reported `OK`.

## Disposition

Keep both tails separate and default-off. Use `--fine-detail-pursuit` when the explicit objective
is sparse deep fabric/detail-band correction; retain `--fine-detail` when broad foreground and
boundary error is the objective. Any default or efficiency decision needs fresh images,
replicated seeds, and count/rate/work-matched controls.
