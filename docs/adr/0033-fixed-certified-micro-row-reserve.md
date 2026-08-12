# ADR-0033: Fixed certified micro-row reserve during local recovery

- Status: accepted (experimental interface); maintained defaults unchanged
- Date: 2026-08-12
- Tasks: HIER-031
- Related: ADR-0017, ADR-0019, ADR-0022, ADR-0030, CORE-010/011, FIT-040

## Context

Mask containment with a positive margin and the ordinary 0.35-pixel scale floor can make valid
foreground pixels unrepresentable. On HIER-031's exposed 1200x1038 C0001 mask, three connected
components containing ten pixels have no legal ordinary Gaussian centre. Increasing the number of
ordinary rows cannot cover those components while retaining the same containment certificate.

A 0.08-pixel isotropic row has certificate radius `0.75 + 3 * 0.08 = 0.99` pixels and can be
placed at every active pixel of this particular binary mask. Such a row must remain smaller than
the fitter's ordinary 0.35-pixel lower clamp, and its independently certified position must not be
reprojected through a constraint configured for ordinary rows.

## Decision

1. `fit(..., constraint_exempt_row_mask=...)` may preserve an independently certified fixed row
   cohort through the ordinary scale floor and mask projection.
2. Every exempt row must also be frozen by `trainable_row_mask`. The fit rejects a missing row
   mask, an exempt/trainable overlap, an empty or wrong-length exemption, and use with
   `active_row_count`.
3. Exempt fitting must be topology-free. Triage, pruning, splitting, relocation, boundary birth,
   and adaptive count are rejected because they can change row identity and invalidate the
   exemption-to-certificate correspondence.
4. The caller owns the independent containment certificate. The hook does not infer that a row is
   safe from its scale alone and does not relax the exact outside-zero endpoint checks.
5. The hook is an experimental recovery interface, not a new stored field attribute. HIER-031
   materializes its endpoint as the ordinary four arrays only; no exemption mask, SDF, support cap,
   or auxiliary mask is persisted.

## Consequences

- A small fixed cohort can guarantee representability of thin mask topology while ordinary rows
  continue to optimize appearance.
- The 0.08-pixel value is source/protocol-specific evidence, not a new global minimum scale or CLI
  default. Other margins, cutoffs, sampling conventions, and masks require a fresh certificate.
- Freezing micro rows makes the local recovery intentionally asymmetric. Count-neutral donor
  funding and any later terminal closure remain method-level decisions and are not part of
  `fit()`.
- HIER-031's selected exposed endpoint is encouraging but not promotion evidence: it uses one
  image, seed, and device, and remains softer than the ordinary pipeline despite eliminating raw
  holes.

## Development evidence

HIER-031's selected exact-7,000-row endpoint uses 910 fixed micro rows and 6,090 ordinary rows. It
has zero raw foreground coverage holes and exactly zero support/reconstruction outside the mask.
Relative to the frozen HIER-030 cold additive control, it gains 2.2844 dB overall, 2.3560 dB in the
four-pixel boundary band, and 0.7546 dB in the deeper interior while reducing deep high-pass MSE
6.31%. The untouched fixed-capacity pipeline is sharper but leaves 933 raw holes. These are
dirty-source, exposed C0001 diagnostics; `formal_claim_ready=false` remains binding.

Evidence: `ara/evidence/hier031-exact7k-masked-boundary-detail-2026-08-12/run.md`.
