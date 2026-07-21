# PORT-005: Batch encode throughput (multi-process across images)

**Status: implemented (CPU-validated); GPU scaling measurement pending.** Recommended by the
2026-07-21 CUDA/Thrust port feasibility study
(`docs/research/2026-07-21-cuda-thrust-port-feasibility.md`) as the zero-rewrite-risk path to
production encode *throughput*: per-image fits are independent, so images/hour scales by worker
parallelism, not single-fit latency work.

## Context

`structsplat fit` encodes one image per invocation. Farm-style encoding (many images, one or
more GPUs) previously required external scripting, with no shared metrics stream, no resume, and
no device assignment policy. The sweeps in `benchmarks/` shard experiment cells, but they are
experiment harnesses, not an encode CLI.

## Goal

A `structsplat batch-fit` command that encodes a set of images across worker processes with the
exact `fit` option surface, round-robin device assignment, dynamic load balancing, per-image
failure isolation, resumability, and a reproducible per-image metrics log.

## Acceptance criteria

- [x] `batch-fit` accepts files, directories, and glob patterns; expansion is sorted and
      deduplicated; same-named files from different directories get collision-free output names.
- [x] Workers are separate spawned processes pulling from a shared queue (dynamic load
      balancing); each worker is pinned to one device from `--devices` round-robin.
- [x] Every image writes the same artifacts as `fit` (`<base>_<strategy>.npz` + render PNG) plus
      one JSON row in `<outdir>/metrics.jsonl` carrying metrics, timings, device, seed, and the
      JSON-safe option surface (reproducibility invariant).
- [x] Resume: images whose output `.npz` exists are skipped unless `--force`; interrupted runs
      continue from the on-disk state.
- [x] One bad image yields an error row and a nonzero exit code without killing the run; a hard
      worker death (OOM kill) fails loudly instead of hanging, and a rerun resumes.
- [x] Mask options are rejected with a clear message (mask pairing is per-image `fit` work).
- [x] CPU-runnable tests cover expansion, collision handling, config serialization, a real
      two-worker spawn round trip, resume, failure isolation, and mask rejection
      (`tests/test_batch.py`).
- [ ] Multi-GPU scaling row (images/hour at 1/2/4 workers on fixed inputs) recorded on real
      hardware; near-linear scaling is the expectation for same-size images, degraded by
      stragglers for mixed sizes.

## Interfaces touched

`src/structsplat/batch.py` (new), `src/structsplat/cli.py` (`build_fit_configs` extraction,
shared fit-option parent parser, `batch-fit` subcommand), `tests/test_batch.py`, `README.md`.

## Notes

- 2026-07-21: Implemented as above. Design choices: one shared work queue + per-worker device
  binding (dynamic balancing without per-device queues); parent process is the only
  `metrics.jsonl` writer (no file races); the configured `--seed` seeds every image's init
  identically, matching the ablation harness's explicit-seed practice; `--torch-threads` guards
  CPU oversubscription. CUDA atomics nondeterminism (ADR-0011) applies per fit exactly as in
  `fit`; rows log device and renderer so provenance stays interpretable.

## Depends on

FIT-001, CORE-001, ADR-0011 (renderer provenance), PORT-001 (shares the exact CUDA renderer).
