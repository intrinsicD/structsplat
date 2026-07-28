# FIT-043 sequential error-then-pursuit diagnostic

## Scope

This preregistered task reuses the audited base, FIT-031 error-only, and FIT-040 pursuit-only
outputs for all 51 non-reference Janelle cells and executes only the previously missing
`error_then_pursuit` arm. The cells are correlated views from two adjacent frames of one capture,
with one seed and natural unequal row counts. This is not FIT-042 independent confirmation,
equal-row evidence, actual-rate evidence, or a default decision.

## Frozen controller

1. Run error-only first by loading its exact persisted field.
2. Skip pursuit when error-only already reaches cumulative 25% high-pass and 20% Laplacian
   reduction relative to the original base.
3. Otherwise convert each cumulative target with
   `max(0, 1 - base*(1-T)/stage_entry)` and run the unchanged 128-row FIT-040 waves up to 2,048.
4. Freeze every base+error row and require both the stage-entry and original-base protected gates,
   exact outside zero, and the cumulative detail targets.

## Result

- Completed: 51/51; failures: 0.
- Pursuit executed: 44; skipped as already satisfied:
  7.
- Cumulative target hits: 51/51.
- Original-base protected safe / outside exact zero / executed prefix exact:
  51/51 /
  51/51 /
  43/44.
- Median incremental pursuit rows:
  256;
  median total combined tail rows:
  6,272.
- Median cumulative high-pass/Laplacian reductions:
  27.30% /
  28.27%.
- Median combined foreground-PSNR gain:
  +3.619718 dB;
  median change from error-only:
  +0.048888 dB.

Frozen decision: **reject the frozen sequential controller without retuning**. All four rules:
`False`,
`True`,
`True`,
`True`.

Even on a pass, the result supports only the feasibility of separate ordered stages on this
exposed capture. Pursuit-only remains the row-efficient fine-detail arm; error-only remains the
global-fit arm. A production opt-in combination needs a later interface task and ADR amendment.

## Provenance

- Frozen input: `/home/alex/Documents/structsplat/runs/janelle_cross_view_tail_diagnostic_20260728`
- Input hashes: `{'manifest.json': 'f8958a583c238cf649b9662b84130d75d2bf3afad7760f43ce22da9245ee6976', 'summary.json': 'c1cafe794fc73e8e32212d00b4edc9e8e48fe863d71c97195f8bd9db64bcc638', 'comparison.csv': 'bdde1bd87adba10bd9ee1e33bc189c963d597a4fc81ae4311739d85c4a24c564'}`
- Raw output: `/home/alex/Documents/structsplat/runs/fit043_sequential_error_pursuit_20260728`
- Command: `['/home/alex/miniconda3/bin/python', 'scripts/experiments/fit043_sequential_error_pursuit.py', '--quiet']`
- Environment: `{'python': '3.11.15 | packaged by conda-forge | (main, Mar  5 2026, 16:45:40) [GCC 14.3.0]', 'platform': 'Linux-6.8.0-117-generic-x86_64-with-glibc2.35', 'torch': '2.7.0+cu126', 'torch_cuda': '12.6', 'device': 'cuda:0', 'gpu': 'NVIDIA GeForce RTX 4090', 'pillow': '11.0.0', 'renderer': 'cuda', 'git_commit': '1da0e68d24124f88a71bda173793e667aa88aa47', 'git_dirty': True}`
- Executed source snapshot: `[{'path': 'scripts/experiments/fit043_sequential_error_pursuit.py', 'sha256': '3913f318781308bfbf03d34918b9cd7f101d6346452ad3b5383e973765aec0e7', 'bytes': 59765}, {'path': 'scripts/experiments/audit_fit043_sequential_error_pursuit.py', 'sha256': '23346e930cb10a9b714ea3af61fdabf08e7ce0a7dcc91ae669a66ef1272037ed', 'bytes': 20501}, {'path': 'scripts/experiments/run_janelle_cross_view_tail_diagnostic.py', 'sha256': '883c10f5d30bafe15ad855ce1fdb9a91451ff350067e7b90c890505a62274654', 'bytes': 62459}, {'path': 'scripts/experiments/fit032_janelle_dipole_screen.py', 'sha256': 'b6892fb446e5a376d63be2decfd185ed0e7bdd83c627484c6ee239349c0bd844', 'bytes': 33235}, {'path': 'scripts/experiments/fit033_janelle_highpass_solve.py', 'sha256': 'cbe9b13aaf860be29641d037d65de4df90a66831936751262fadd1bb2e5b1543', 'bytes': 27530}, {'path': 'scripts/experiments/fit040_janelle_production_pursuit.py', 'sha256': '64a7407b2d28cead2d8dea97ac1627ba54eb106d350122c7ea4986d22655d772', 'bytes': 12423}, {'path': 'src/structsplat/detail_pursuit.py', 'sha256': '79b5be75fc16b7318a87aef35ed842ba92cd37144fca540497236fcd53c52c30', 'bytes': 15455}, {'path': 'src/structsplat/safe_schedule.py', 'sha256': '98650958a57fbd89d2a6cbacacfcd8c11b98588a3a3292a0ef01a7c8bed69854', 'bytes': 158833}, {'path': 'tests/test_fit043_sequential_error_pursuit.py', 'sha256': 'cb43dd42fd8a126873ec8931a5a218b7716c7bcb0dad9b2d2a616ad309b48328', 'bytes': 2160}, {'path': 'tasks/FIT-043-sequential-error-pursuit-tail.md', 'sha256': 'd8962279ef12d6de91c1350620cb61043fc0dfe9327ebdfef4733800560a743c', 'bytes': 7575}]`

## Reproduction

```bash
PYTHONPATH=src:. python scripts/experiments/fit043_sequential_error_pursuit.py --quiet
PYTHONPATH=src:. python scripts/experiments/audit_fit043_sequential_error_pursuit.py
```
