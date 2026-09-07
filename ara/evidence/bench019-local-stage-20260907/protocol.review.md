# Independent StructSplat phase-two protocol review

Reviewer: Codex-cross-repo-review
Driver: Codex-bench019-driver
Self-reviewed: No
Verdict: Approved
Design SHA256: 00b85f3b7eab01d82c4e96215052670260535b7017d4791454562213223f41b6
Reviewed protocol file SHA256: 82735a7ec0a948306f8c8d58f105dd0d019b875b1f4640c5a99ec06eaf8bab9a
Outcome access: No downstream/held-out outcomes. Stage-1 generated inputs were inspected only for contract, finite-value and provenance checks, with no ranking, selection or scientific choice changes.

## Reviewed envelope

The exact review-state protocol binds ready RTGS task SHA256 8a1f27a7b259b496df28e41b0ebd218536dab40920278f6908865a45dcfae6f0 and approved RTGS protocol SHA256 0df467f6418273465167297edc866c7d526c78aac11e8c0f97fe6e409f7cf2c5. Both source trees were clean: RTGS f165d353ff9e9174c1ac1c936ed7855cc7a25abb and StructSplat 6ff898e8682cc7d932d540b818b3f0d6a4a0e529. Existing coordinator guard and both internal/external source envelopes passed.

Generated-input inventory SHA256 c69aaf7b641a4944bf0eb9e8ead363dc63a73ae0e2e5c2301391d4ce761e1d62 covers 144 files. The current complete input inventory matches it. All six family manifests/metric sources match the protocol descriptors; every one of the 48 field archives cold-loads with 512 original/final rows, the expected provider/blend/kernel/support/epsilon/crop semantics, and no mask-derived bounds. Exact per-view semantic definitions were independently reconstructed and their digests match the protocol. Consumed bytes and row counts match the declared metric metadata.

For each frame/view, all three families share exact target, soft-mask, crop, camera and predictor-sample identities. The per-view fitted configuration and seed match the ready task, source provenance matches the live producer envelope, and recorded cold-replay errors satisfy the predeclared tolerance. Predictor checks only established the declared schema, training-only inventory and finite values; no predictor magnitude, ordering or fit-quality conclusion was used.

All 44 selected RGB/mask source descriptors and two calibration indexes match the original raw seal and source bytes. The indexes preserve training/held-out roles and explicitly contain descriptors rather than pretend JSON contains images. No source images or masks were decoded by this reviewer.

## Scientific design and controls

Task/capture identity, workload-specific scope, two frames, three families, splits, seeds, exact downstream command/schedule, predictor signs and priority, response signs/primary metric, analysis floors and A/A identities/tolerances match the already-approved RTGS task. No post-acquisition scientific choice remains. Field equation differences are explicit. The extra contained A/A remains a local gate/evidence record alongside the native portable A/A. Resource directions required by the portable schema remain descriptive; only held-out foreground PSNR is primary.

The comparison remains one-capture development at one fixed capacity/horizon. Broad surrogate validity is not evaluated because the capture minimum is unmet. A/A failure, missing cells or integrity drift fail closed, with no tolerance tuning, regeneration or default promotion. Fixed paired materiality, alpha guard and initializer-fallback interpretation remain exactly as frozen.

## Evidence and boundary

Existing StructSplat protocol validation, RTGS review-state identity validation, coordinator source/raw checks and independent reconstruction of the field/index bindings all passed. Machine details are in structsplat.independent-checks.json. The downstream root and canonical frozen filename were absent both before and after the review. No downstream command, finalization or canonical publication was performed.

Approval is for this exact design. The driver must finalize to a staging filename, commit the required protocol/review/task metadata without executable changes, preserve initial/new clean commit provenance, verify both source envelopes, and only then atomically publish the canonical frozen protocol. Final results still require independent raw-outcome audit and both report/browser gates.

## Copyable owning-task review block

### Protocol review

#### Reviewer
Codex-cross-repo-review

#### Verdict
Approved

#### Protocol digest
00b85f3b7eab01d82c4e96215052670260535b7017d4791454562213223f41b6

#### Digest scope
BENCH-019 local Stage phase-two design, computed by benchmarks.stage1_downstream_objective.design_digest from the exact reviewed/finalized protocol. The digest excludes only lifecycle/review seals (state, design_sha256, protocol_sha256, review) and binds all scientific settings, clean source provenance, generated field/metric descriptors, raw-source indexes and immutable RTGS task/schedule. The committed frozen copy and independent review are retained under ara/evidence/bench019-local-stage-20260907/; the identical run copies remain under the canonical RTGS run/protocol directory.

#### Outcomes accessed
No

#### Review focus
Exact source/input provenance, 48 cold field contracts, common targets/cameras/samples, training/held-out isolation, unchanged scientific choices, A/A and missing policies, shared resource accounting, capture insufficiency, and staged publication before any downstream outcome.
