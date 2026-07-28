# Janelle cross-view fine-detail tail diagnostic

## Scope

This user-requested diagnostic compares FIT-040 orthogonal pursuit with FIT-031's
error-only tail on all persisted Janelle image/view cells except the previously used
`frame_00008/C0001` reference. It is descriptive evidence within one capture session,
not FIT-042 independent-scene confirmation.

## Frozen paired protocol

- Cells: 51 automatically discovered remaining
  frame/view pairs across frame_00008, frame_00009.
- Shared base: the integrity-checked `.rtgsv` field paired with each calibrated RGB and
  authoritative packed mask, adapted to max-side 1200.
- Renderer: `cuda` for both arms.
- FIT-040 arm: shipped 128-row pursuit waves, 2,048-row ceiling, 25% high-pass and
  20% Laplacian stopping targets.
- FIT-031 arm: shipped residual-support estimate at fraction 0.5,
  512-row allocation waves, and the natural 4,000-step convergence ceiling.
- Shared target rule: protected-safe plus at least 25% deep sigma-1.5 high-pass and
  20% deep Laplacian reduction. If both pass, fewer added rows wins.
- Seed: 0. No repeated-seed inference is claimed because the methods are
  deterministic under this setup.

## Execution

```bash
/home/alex/miniconda3/bin/python scripts/experiments/run_janelle_cross_view_tail_diagnostic.py --quiet
```

- Git commit: `1da0e68d24124f88a71bda173793e667aa88aa47`
- Dirty worktree recorded: `True`
- GPU: `NVIDIA GeForce RTX 4090`
- Torch / CUDA: `2.7.0+cu126` /
  `12.6`
- Pillow: `11.0.0`
- Source snapshot: `[{'path': 'scripts/experiments/run_janelle_cross_view_tail_diagnostic.py', 'sha256': '883c10f5d30bafe15ad855ce1fdb9a91451ff350067e7b90c890505a62274654', 'bytes': 62459}, {'path': 'scripts/experiments/fit032_janelle_dipole_screen.py', 'sha256': 'b6892fb446e5a376d63be2decfd185ed0e7bdd83c627484c6ee239349c0bd844', 'bytes': 33235}, {'path': 'scripts/experiments/fit033_janelle_highpass_solve.py', 'sha256': 'cbe9b13aaf860be29641d037d65de4df90a66831936751262fadd1bb2e5b1543', 'bytes': 27530}, {'path': 'scripts/experiments/fit040_janelle_production_pursuit.py', 'sha256': '64a7407b2d28cead2d8dea97ac1627ba54eb106d350122c7ea4986d22655d772', 'bytes': 12423}, {'path': 'scripts/experiments/fit041_janelle_equal_base_error_tail.py', 'sha256': 'c38d9bb2ecc5d7a4e12a512595e6159fcad5e526b2e297694a4d84c22adc0953', 'bytes': 11315}, {'path': 'benchmarks/highpass_births.py', 'sha256': '4a86223116ebf539792de7773844fbdc6239be7234cf0a77f256bee663f2839a', 'bytes': 6194}, {'path': 'benchmarks/residual_birth_color_solve.py', 'sha256': '44709267137a7495d2c62e7d3acf23531d65c911664fa85a27638bc755d8745b', 'bytes': 5301}, {'path': 'src/structsplat/detail_pursuit.py', 'sha256': '79b5be75fc16b7318a87aef35ed842ba92cd37144fca540497236fcd53c52c30', 'bytes': 15455}, {'path': 'src/structsplat/safe_schedule.py', 'sha256': '98650958a57fbd89d2a6cbacacfcd8c11b98588a3a3292a0ef01a7c8bed69854', 'bytes': 158833}, {'path': 'tasks/FIT-031-error-only-fine-detail-tail.md', 'sha256': '6dac593ecea9be1b222658b44e90a66dc59d516885e4f1a12808a1856296d14c', 'bytes': 4378}, {'path': 'tasks/FIT-040-opt-in-orthogonal-detail-pursuit-tail.md', 'sha256': 'd514e66be6c83ce5c70252991b1af32164f419bd937b803463e459548db214bb', 'bytes': 3293}, {'path': 'tasks/FIT-041-equal-base-error-tail-control.md', 'sha256': 'ac64842738bb9e134c2962ccd068c1af8f7b055b76c05766fb3336878ada6b7a', 'bytes': 2726}, {'path': 'tasks/FIT-042-independent-fine-detail-pursuit-confirmation.md', 'sha256': 'e48e7ba102e230f3f27620da1f736591590bc7fa948b27cd8314d8cb68793dbe', 'bytes': 11323}]`

## Descriptive outcome

- Completed / eligible: 51 / 51
- Pursuit target hits: 51/51; median added rows
  384; median high-pass / Laplacian reductions
  26.7% /
  26.9%.
- Error-only target hits: 7/51; median added rows
  6,144; median high-pass / Laplacian reductions
  12.8% /
  14.8%.
- Winner counts: pursuit 51, error-only
  0, tie 0, neither
  0.
- Failures: 0.

## Limitations

1. Views from the same calibrated capture and two adjacent frames are strongly correlated.
2. The bases are archived byte-capped, mask-contained fields (roughly 5k rows), not fresh
   current-pipeline 10–11k fits. The comparison is internally paired but does not reproduce
   the exact prior C0001 base distribution.
3. The arm budgets are natural method budgets, not equal-row budgets. FIT-031 is expected to
   favor broad foreground PSNR; FIT-040 is explicitly optimized for the fine-detail target.
4. Pillow 11.0.0 materialized these targets. Exact target and mask PNG hashes are
   recorded per cell.
5. This evidence must not be counted as FIT-042's sealed independent-scene screen or
   confirmation set.
