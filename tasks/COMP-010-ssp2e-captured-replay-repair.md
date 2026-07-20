# COMP-010: SSP2E captured-replay relocation repair

## Status

**Completed and independently audited GO for captured-source codec replay.** Repair protocol v2
was specified after COMP-009 analysis and two read-only failure-localization audits. Protocol v1
allowed only asset-path relocation. A source-only draft of that launcher was
written, but no tests, preflight, child execution, or result artifact were run. The independent
audit then proved v1 insufficient because the non-gating CUDA renderer proof also depends on a
mutable, non-reproducible JIT cache. Protocol v1 is retired without a decision; v2 froze both
exceptions before its first real attempt.

The visible frozen COMP-009 decision is `ABANDON_FIXED_SSP2E_V1`; this repair may validate that
unchanged codec decision or yield lifecycle no-decision. It may not change a persisted codec
stream, model, timing sample, row, gate, threshold, bootstrap sample, or decision. The captured
worker necessarily regenerates candidate streams ephemerally and must discard them after exact
comparison; describe that honestly rather than claiming that no stream computation occurred.

## Failures being repaired

COMP-009 completed its preflight, 16-cell run, resource benchmark, and frozen analysis. Its
captured-source replay stopped inside `verify_preflight` before deterministic codec replay because
`assets.verify_assets()` returned absolute paths rooted in the temporary extraction, while the
preflight record contained the original repository-rooted absolute paths. Asset content was
already identified by exact size/hash and the extracted source archive was already identified by
its exact manifest/hash. Treating those host paths as scientific identity made an intentionally
relocatable replay non-relocatable.

After an exact allowlisted simulation of those two asset paths, the audit reached a second failure
in `verify_renderer_runtime`. The preflight renderer extension and the captured rebuild had the
same cache path, 397,376-byte size, exact ABI/readelf records, and exact toy-output hash
`f53ab7de7f05942b410c106dcb7a24cdf02d4c082196e765491773e6f70725e4`, but their whole-file hashes
were respectively `8ad8a640ad9af29fce5c2d4841fc85495ebe6fd1f5c22ae13950fd454b6c6075`
and `adaa5ecf87c69b935bd55d2f8b6b0b2080a686d512152282576aacce2df89a9f`.
Independent original-root rebuilds produced still other hashes despite byte-identical source and
the frozen environment. Inspection showed embedded extraction paths, NVCC `tmpxft_*` names, and
ELF build IDs. The exact preflight renderer binary was not persisted, so its hash cannot be
recreated or reloaded. The renderer was explicitly scoped as a render-only, non-gating diagnostic
and is not invoked by deterministic codec-cell replay.

These are lifecycle harness errors, not failed rate gates. Until this separately bound repair
passes, COMP-009 remains lifecycle-incomplete and its negative codec decision is not final. Even a
passing repair cannot claim exact renderer-binary replay.

## Transparent v2 attempt history

The first v2 preflight, in
`results/comp010_ssp2e_replay_repair_v2_2026-07-16`, passed with seal
`e41c2094241e72c525f1fd6c665453ec55e0d4e31658ae4ee48dd5bf09f6e86f`. It bound repair source
`dafb14f5ed9ba39a1e3bd5f521ff9597318a0c46b472d47a61543881a7f4a9bf` and hostile tests
`7d73c9708b3cbd667d9864d73680d14d680844dc0650d9abfd1c28dd46036c3a`. Its first captured child
completed the 16-cell worker but the parent failed closed before writing `repair.json`: under
`python -m`, the child entry existed as `__main__`, so both module-origin tables omitted the
canonical logical `benchmarks.ssp2e_replay_repair` record required by the parent. Every other
origin record matched. This was a launcher provenance-recording error, not a codec mismatch; the
attempt has lifecycle no-decision and its artifact remains unchanged.

Before a fresh retry, the origin guard was made to add that one logical alias only when
`__main__.__file__` resolves to the manifest-verified extracted child entry. A regression test now
reconstructs the real `python -m` table and requires exact before/after origin records, a single
canonical child identity, no `__main__` scientific identity, and no mutation of the supplied module
table. The repaired source is
`1319160115304551e9662b9e142fed1138453fa5bf78d9de285d99d15f4b13ff`; the 81-test suite is
`dde04c8296fc9f3c87d106bbb56daae606e61a0e369f9135c6054d33c400b00b`. These hashes and this
updated task must receive a new empty-output preflight and independent audit before retry. No
scientific exception, codec field, gate, or persisted COMP-009 record changed.

The fresh retry in
`results/comp010_ssp2e_replay_repair_v2r2_2026-07-16` passed preflight with seal
`a00a7a553468dbe1f2e00e1d6e5ecb38fb3227356e090c73bbf564b39ec1bc00` and completed with repair
seal `adbf2c48e1721b6d4b74960211e08ec3c770bebde1d068ddfc88e5a83f78f3a8`. Two distinct randomized
extraction roots produced the same child seal
`b482a866742d472d02d7c80db1c86640e5e17a85fdf025a56096173c08fd7ac4` and captured-worker seal
`0d7d74a8e6f6a552eeb4f1d3df4a0aeb8f6e6977496400d7c65845c4fd494309`. Independent post-result
audit recomputed all 16 ordered input identities, all 16 normalized non-timing execution hashes,
and all 64 complete-stream hashes; verified the live/before/after 828-file COMP-009 inventory as
identical; and returned GO with no P0--P2 finding.

This repairs the frozen negative codec decision's captured-source provenance only. The renderer
runtime and exact renderer binary were not replayed, persisted resource timings were reused rather
than remeasured, and no new quality, performance, convergence, expressiveness, or compression
evidence was created. The artifact proves no persisted COMP-009 mutation; it does not claim that a
perfectly reverted direct-native or pre-opened-descriptor transient write was physically
impossible.

## Frozen inputs

Use only `results/comp009_ssp2e_actual_dev_v1_2026-07-16` with:

- preflight binding `b4d843ce22356839fa3fa39082044e61cc7ee82cc71176780a192d811d63fadc`;
- source manifest `d75ee551ee650f0d465a01fc8f49b2ee0c4eee15579cf362ee3dfda2b96ad7d6`;
- source archive `fac4ca0978891b3cd16d477ffc04d12dc120168478192293d39af635fca7eb50`;
- input manifest `8f12a64d484f4239a121277ff0467409ed374856ddbf134dfaf49c737fea5f1a`;
- run manifest `cc21fc84aab5cd55ecfffc358f67782013d7a7cf56ffc9b8ef34f8ea98de0a24`;
- benchmark manifest `88a706da64836544d997b177b3b4dda2610e6ede3eceb457e98a4aa7899fe162`;
- analysis record `435f011fe598263cd304b1cbe0754ca82e5fe1e7a49536db4099bdfff9166201`;
- analysis `f10c4a1906e4bd240c10253508f38c4053cfe332dec13b220262fee7ae990b30`;
- frozen decision `ABANDON_FIXED_SSP2E_V1`.

The failed original replay produced no `replay.json`. Do not open COMP-008 targets, fields, NPZs,
pixels, or any confirmation material.

## Exactly two allowed lifecycle exceptions

After extracting and verifying every file in the frozen source archive, load the frozen
`benchmarks.ssp2e_actual_run` module from that extraction.

### 1. Asset relocation

Wrap its imported `ssp2e_assets.verify_assets` function:

1. call the original extracted function on the extracted asset directory;
2. require exact equality with the preflight asset record for `cdf_sha256`, `cdf_size`,
   `cost_sha256`, `cost_size`, `alphabets`, and `cost_count`;
3. require the observed `cdf_path` and `cost_path` to resolve to the two manifest-verified files in
   the extraction; and
4. return the same observed record with only those two nonsemantic path strings replaced by the
   corresponding frozen preflight strings.

The wrapper must be single-purpose and fail on any other difference.

### 2. Sealed non-gating renderer-proof reuse

Wrap only the captured runner's `renderer_runtime_proof` function so the original captured
`verify_renderer_runtime` still executes unchanged:

1. require the renderer record to be the exact record in the hash-verified COMP-009 preflight,
   including scope, device, 397,376-byte extension identity, ABI, and toy values/hash;
2. independently require the captured manifest hashes for `cuda_render.py`, `render.py`,
   `gaussians.py`, `render_ext.cpp`, and `render_ext.cu`, including C++ hash
   `eec0e011189a34d3489c6f719d15528a3511758b79b5cfad145f91323ef5e356` and CUDA hash
   `5493130d56cdf24dd67d54111eb14a701f5e2b143173cdee27a90a7fa988e150`;
3. return an isolated copy of the exact sealed renderer record rather than building or loading a
   renderer extension;
4. count every call and record that the renderer proof was reused, not reexecuted; and
5. install fail-closed guards proving that renderer JIT build/load was never attempted.

This is an explicit scope exclusion, not evidence of renderer reproducibility. The repair result
must say `renderer_runtime_reexecuted=false` and `exact_renderer_binary_replay_claimed=false`.
No other function, constant, record, source file, environment field, module origin, or artifact may
be patched, normalized, or reused in place of execution. Both wrappers and all guards must be
restored in a `finally` block.

## Source and execution binding

Before running the repair, create an empty repair output and bind this task, the repair module and
tests, the exact COMP-009 seals above, Python/environment/loader state, and the repair source
hashes. Reverify them immediately before execution.

Extract the frozen COMP-009 source archive into a fresh directory; reject absolute paths,
traversal, links, unmanifested files, missing files, or any size/hash drift. Place an exact copy of
the source-bound repair module into the extraction solely as the child entry point, record that
addition explicitly, and set `PYTHONPATH` to the extracted root plus `extracted/src`. Require all
COMP-009 benchmark modules to originate under the extraction and all `structsplat` modules to
originate under `extracted/src`.

Reuse the exact frozen loader/thread environment. The child must invoke the captured
`captured_replay_worker`; it may not call a live COMP-009 implementation. The exact persisted
native arithmetic coder `5e7ae9813b5d56b51bdba2475268cb9158814f214cb34d6f390f8242f1de336a`
must be loaded and its frozen ABI/proof reverified; it must never be rebuilt.

## Acceptance

The repair passes only if the captured worker:

- reproduces all 16 ordered image/tuple identities;
- reproduces every non-timing execution/model/blob hash exactly;
- revalidates the stored raw timing samples without rerunning or replacing them;
- recomputes the exact analysis seal and `ABANDON_FIXED_SSP2E_V1` decision;
- reports no field/QAT refit, pixels, or confirmation access;
- has exact captured runner/source-manifest/run-manifest/analysis-record bindings; and
- exercises only the two exceptions above, with active asset parsing, sealed renderer-proof reuse,
  no renderer JIT load/build, and both original captured verifiers otherwise unchanged.

Run the repair twice from independently randomized extraction roots. After removing only explicitly
recorded random extraction paths from the repair launch envelope, both captured worker records and
all scientific identities must be byte-identical. Inventory hashes before and after must prove that
no COMP-009 file changed. Ephemeral replayed streams may exist only in child memory; no stream,
timing, or analysis payload may be written to COMP-009.

Any failure is lifecycle no-decision. A pass validates captured-source **codec** replay and repairs
the negative codec decision's provenance only. It does not validate renderer-binary replay,
strengthen the compression claim, rescue SSP2E v1, authorize tuning, or authorize confirmation.

## Required hostile tests

Before repair preflight, reject mutations of every frozen input seal, either asset hash/size/table
identity, either extracted asset target, an extra normalized field, renderer scope/source/toy/ABI
record, any renderer JIT attempt, archive traversal/link/file drift, live-module leakage,
environment drift, any cell/blob/model identity, timing replacement, analysis/decision drift, or
pixel/confirmation flags. Boolean success flags never substitute for exact hash equality. Tests
must also reject 15/17/reordered/duplicate cells, mutations in each of the four complete-stream
hashes, normalized execution identities, model/grid/head selections, native-coder identity, and
coherently resealed but substantively mutated fixtures.

## Interfaces allowed

One new benchmark repair module, its tests, this task, ignored repair evidence, research docs, and
ARA records. Do not edit the frozen COMP-009 task/source archive/artifact, production source, or
COMP-008 material.
