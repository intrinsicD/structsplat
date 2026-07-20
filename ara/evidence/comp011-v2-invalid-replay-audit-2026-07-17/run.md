# COMP-011 v2 replay and claim audit

**Date:** 2026-07-17
**Scope:** adversarial results audit of development artifact
`results/comp011_ssp2v_actual_dev_v2r36_2026-07-17`
**Verdict:** `INVALID_NO_DECISION`; no confirmation authorization

## Decisive finding

The ordinary lifecycle completed and emitted
`ABANDON_FIXED_SSP2V_V1_UNDER_COMP011_V2`, but the frozen task makes a captured-source replay part
of validity. Replay phase A reproduced the complete candidate set and cold decoded states. All 16
fresh replay quality schedules then completed, after which the captured quality reducer correctly
rejected the evidence because a decision-relevant original metric was already inside an exclusion
band.

The unique original ambiguous constituent among the 375 metric classes in the 125
decision-relevant prefix arms is:

| Field | Value |
|---|---|
| Image | `vita-vilcina-3055` |
| Tuple | `[16,8,8,8]` |
| Selected exact comparator | `ssp2f` |
| Prefix arm | `flat_s2_k128_l384` |
| Byte-order position | 2, before selected `rvq_s3_k64` at position 6 |
| LPIPS increase | `0.00101422518491745` |
| Frozen LPIPS exclusion band | `(0.00098,0.00102]` |
| PSNR loss | `0.9610763099751942` (`definitely_fails`) |
| MS-SSIM loss | `0.004414200782775879` (`definitely_fails`) |
| Aggregate arm class | `definitely_failed` |

The same LPIPS value and ambiguity occur in ordinary v2r35 and v2r36. This is not replay noise.
Even a numerically identical replay must fail because the verifier checks both the original and
replay per-metric classes. Task lines 678--680 require every decision-relevant replay metric to
remain outside every exclusion band; lines 998--1006 make this arm part of the replay prefix; and
lines 919--920 define any replay failure as invalid/no-decision.

Changing the verifier to accept an aggregate `definitely_failed` arm after seeing the in-band
constituent would relax a frozen rule after outcome access. No threshold, band, prefix, or
classification rule was changed.

## Claim disposition

| Claim | Scope | Disposition |
|---|---|---|
| COMP-011 validates abandonment of fixed SSP2V v1 | Development scientific decision | **Refute.** Full validity failed, so the ordinary abandonment payload cannot be promoted. |
| Ordinary v2r36 produced no promotion signal under its local analysis | Development diagnostic | **Narrow.** SSP2V complete-byte GM ratios were `0.9930826` and `0.9952001`, with `3/8` strict wins at both tuples; all four primary rate gates failed. Selected-V/Q quality and decode-resource gates passed ordinarily, but replay never authorized resource reproduction. These are unreplayed diagnostics, not evidence against VQ or SSP2V. |
| Replay phase A reproduced complete candidate bytes and decoded states | Codec/lifecycle plumbing | **Confirm narrowly.** Phase-A seal `0da5a856d8a7b4b931439eda65c54bcf993806ec342c6eda1edf87260f5ff2f9`; regenerated candidate-set hash `cb27ed8918d79b9c4f0e89c05d0e98a57fbfb266777011d421cb33f959b9caaf`. This does not validate target quality or final decisions. |
| SSP2L development screen | Outcome-informed secondary diagnostic | **Narrow.** Ordinary GM ratios were `0.9568742` and `0.9658907`, with `8/8` rate wins, but decode upper-median ratios were `5.21954x` and `23.97430x` versus the `1.25x` gate. The ordinary screen failed; the enclosing invalid task issued no promoted decision or confirmation authorization. |
| Confirmation data may be opened | Confirmation authorization | **Refute.** No COMP-011 confirmation path is authorized or accessed. |

The ordinary descriptive rate values may remain in the ignored raw artifact, with the explicit
label `invalid/no-decision development diagnostics`. They must not enter a positive or negative
compression claim, default decision, abstract, or confirmation trigger.

## Integrity and execution scope

- Ordinary preflight, candidate generation, quality, and analysis were source-bound.
- Candidate fitting/coding and host reducers used CPU; quality rendering and metrics used the
  persisted CUDA authority on an NVIDIA GeForce RTX 3050 (compute capability 8.6), bound in
  `preflight.json`.
- Replay phase A and all fresh quality workers ran; no phase-B quality authorization, resource
  replay, final replay record, or replay seal exists.
- Confirmation payloads were not accessed.
- v2r36 relevant seals: quality
  `f4e27457fd6322e24dccb298a7985ee2efa2f927f618e4394ba406479f2b9b9c`,
  ordinary analysis record
  `b53e8c1e7d873bc1c7fa75f11c516ef4600780813beba7df471ddf3f728893b8`,
  and replay phase A
  `0da5a856d8a7b4b931439eda65c54bcf993806ec342c6eda1edf87260f5ff2f9`.

## Clean next action

Stop COMP-011 replay reruns: unchanged source cannot clear an original frozen-band ambiguity.
Preserve the invalid run and its negative provenance. Any successor must have a new task identity,
frozen controls and gates, target-blind selection of genuinely new development images, an unopened
confirmation roster, and no tuning on the COMP-011 cells. The reviewed successor direction is an
unchanged-decoder SSP2F exact-complete-byte RGB coordinate-RDO assay; its pre-freeze blockers are
recorded separately in `ara/evidence/comp012-prefreeze-hostile-review-2026-07-17/run.md`.

## Audit reproduction

The decisive source value can be recomputed without opening any upstream or confirmation payload:

```bash
python - <<'PY'
import json

path = "results/comp011_ssp2v_actual_dev_v2r36_2026-07-17/quality_stage.json"
stage = json.load(open(path))
for cell in stage["cells"]:
    order = cell["selection"]["ordered_labels"]
    selected = cell["selection"]["selected_label"]
    prefix = order[: order.index(selected) + 1]
    for label in prefix:
        quality = cell["selection"]["quality"][label]
        ambiguous = [
            metric
            for metric, state in quality["metric_classes"].items()
            if state == "ambiguous"
        ]
        if ambiguous:
            print(
                cell["image_id"],
                cell["bit_tuple"],
                cell["q_name"],
                label,
                ambiguous,
                quality["conservative"],
            )
PY
```
