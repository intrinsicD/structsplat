# Frame 00008 three-arm live conversion audit

## Scope

Read-only forensic audit at `2026-08-03T00:46+02:00` of the active conversion under:

`/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008`

The conversion protocol classifies the outputs as `production conversion; diagnostic comparison
only`. No fit was launched, stopped, resumed, or modified by this audit. At the snapshot,
GaussianImage and StructSplat no-boundary were complete (26/26); StructSplat mask-contained was
6/26 complete and fitting `C0012` on one RTX 3050 worker.

## Sources and binding

- `gaussians2d_11k_protocol.json` and `gaussians2d_11k_verification.json`.
- Every JSON receipt and history under the three `gaussians2d_*_fullres` directories.
- The prior `gaussians2d/fitting/*.json` contained run.
- Bound StructSplat sources `fit.py`, `init.py`, `pipeline.py`, `render.py`, and
  `safe_schedule.py`; their current SHA-256 values match the receipt bindings.
- The custom `convert_three_arm_11k.py`; its current SHA-256 matches the receipt binding.
- QA reconstruction crops for `C0001` in all three arms.

The current protocol file hash differs from early receipt bindings because horizon selection wrote
the selected GaussianImage iteration count back into the protocol. The executed source/config
digests remain present in every receipt.

## Recomputed completed-output summary

| Arm | Views | Mean foreground PSNR | Wall hours | Mean min/view | Mean rows | Attempted | Accepted | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GaussianImage additive | 26 | 33.951045 dB | 1.695614 | 3.912955 | 11,000 | 520,000 fixed updates | 520,000 | 100% |
| StructSplat no-boundary | 26 | 30.391925 dB | 25.897576 | 59.763637 | 6,582.154 | 366,142 | 15,895 | 4.3412% |
| StructSplat mask-contained | 6 | 28.947106 dB | 6.381226 | 63.812259 | 11,000 | 98,620 | 9,177 | 9.3054% |

GaussianImage does not label its fixed 20,000 updates as accepted/rejected; the table reports them
as retained trajectory work only to make the absence of rollback explicit.

Paired foreground-PSNR comparisons:

- no-boundary minus GaussianImage, 26 pairs: `-3.559120 dB`, 1/26 wins; the mean dB gap
  corresponds to about `2.269x` foreground MSE;
- contained minus GaussianImage, six completed pairs: `-5.071362 dB`, 0/6 wins, about `3.215x`
  foreground MSE;
- contained minus no-boundary, the same six pairs: `-1.250493 dB`, 2/6 wins.

The no-boundary arm is `15.27x` slower in total wall time than the complete GaussianImage arm.
Wall time per recorded update is approximately `21.7x` larger for no-boundary and `19.8x` larger
for contained, despite StructSplat attempting fewer updates per view.

## Where StructSplat time goes

Elapsed schedule-event attribution excludes initialization/final serialization and assigns each
cumulative-event interval to its phase.

| Phase | No-boundary hours / share | Contained-six hours / share |
|---|---:|---:|
| bootstrap | 1.478 / 5.86% | 0.414 / 6.73% |
| coverage | 7.988 / 31.65% | 1.711 / 27.78% |
| detail | 9.282 / 36.78% | 1.875 / 30.45% |
| general/boundary closure | 3.122 / 12.37% | 0.990 / 16.08% |
| redistribution | 2.663 / 10.55% | 1.026 / 16.67% |
| safe polish | 0.675 / 2.67% | 0.135 / 2.19% |

No-boundary global fitting accepted 10/349 blocks; 339 were fully rejected after up to four
backtracking trials. Top-level rejection reasons included `outside_render_regressed` 302 times,
`outside_coverage_regressed` 254, and `interior_holes_regressed` 75. Mask-contained accepted 7/151
global blocks; its dominant reasons were boundary MSE, CVaR, foreground MSE, and boundary holes.
Safe polish accepted 0/12,168 no-boundary and 0/2,808 contained attempted updates.

## Objective and representation audit

The live StructSplat phase objective is masked RGB L2 plus phase-specific raw-denominator
undercoverage floors. Configuration/code inspection found:

- `ssim_weight=0`, `coverage_match_weight=0`, `geometry_loss_weight=0`;
- `loss_weighting=mask`, constant per-Gaussian RGB, no affine basis, no spherical harmonics;
- structure-tensor `density_mode=structure` is initialization-site density, not a fit loss;
- responsibility/error density ranks topology proposals, not the primary pixel objective;
- the active density-like loss terms are interior and, when enabled, boundary raw-denominator
  coverage floors;
- fixed-geometry constant-RGB color solves occur before bootstrap and before polish, while
  per-event color solves are disabled.

GaussianImage uses an additive compositor and a nine-scalar parameterization consisting of 2D
position, three Cholesky values, RGB, and one scalar amplitude. `weight_color_9p` is not SH. Its
actual masked fit mattes the target and weights outside pixels by `0.1`, whereas StructSplat's
masked pixel term weights them by zero.

The custom no-boundary arm passes the real mask and changes only `boundary_enabled=false`. The
pixel term therefore ignores outside pixels, but the unchanged commit decision still protects
outside render/coverage. The persisted profile also describes authoritative masked containment
although the resolved fit has `mask_contain=false`. This arm is not a clean boundary ablation.

## Convergence audit

StructSplat's persisted `converged=true` means the final transactional phase reached a deterministic
fixed point under the protected metric gate; it does not mean photometric loss reached a local
minimum.

- No-boundary selected mean PSNR is `30.3919 dB`, while the best candidate metric found anywhere
  in each history averages `36.0335 dB`, a `5.6416 dB` gap. Those candidates are unsafe and can
  violate coverage/outside constraints; they only show that the retained field is not a pixel-loss
  plateau.
- In all six contained histories, coverage reached its metric target, detail and boundary closure
  hit their step budgets, and polish reached a deterministic fixed point with zero accepted work.
  The best history candidate averages only `0.0774 dB` above the selected state, so merely running
  the same late phase longer is unlikely to close the roughly 5 dB GaussianImage gap.
- GaussianImage's `C0001` horizon pilot failed the final-500-update gain rule at 5,000
  (`+0.1316 dB`) and 10,000 (`+0.0608 dB`) updates, then passed at 20,000 (`+0.0114 dB` versus a
  `0.05 dB` threshold).

## Earlier frame_00008 directory

The prior `gaussians2d` contained run used a different recipe: full 5,750-row fitting from the
start, normalized `cuda_tiled`, L1 plus `0.3*SSIM`, hard containment, best-full-count checkpointing,
and an approximately 168 kB output cap that serialized 5,318 mean rows. Across 26 views it averaged
23.928334 dB foreground PSNR, 0.868820 foreground SSIM, 0.966693 matted MS-SSIM, and 0.084561
matted LPIPS. It was run on an RTX 4090, so its 2.347 total hours are not a valid current-GPU speed
comparison. Current no-boundary is +6.4636 dB on all 26 views; current contained is +5.2687 dB on
its six completed pairs. Current receipts omit SSIM/MS-SSIM/LPIPS, so perceptual superiority is not
identified.

## Evidence boundary and next discriminating test

This audit is source-bound but not claim-ready: one arm is incomplete, source worktrees were dirty
in receipt provenance, the comparison was labeled diagnostic, the no-boundary control is
semantically inconsistent, and current perceptual metrics are absent.

The smallest causal follow-up should freeze a few representative views and equalize target, mask,
11,000-row count, seed, and either work or wall-clock across: native additive, StructSplat additive
plain fit, normalized plain fit, corrected no-boundary/matted policy, and current contained staged
fit. Only after the renderer and transaction-policy axes are isolated should L2, L1+SSIM, and
Charbonnier or stage-order variants be compared.
