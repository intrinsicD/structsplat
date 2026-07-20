# COMP-009 exact SSP2E actual coder

Fixed SSP2E v1 fails both tuple-level actual-rate and spatial-attribution gates. Actual-rate
geometric-mean ratios are `0.986954` and `0.987294`, with only 6/8 strict image wins and bootstrap
upper ratios `0.999584` and `0.997035`, not the frozen `<0.95` requirement. Modeled-versus-shuffled
aggregate ratios are about `0.9591/0.9616`, short of the required 10% attribution effect. Resource
gates pass, but the conjunctive decision is `ABANDON_FIXED_SSP2E_V1`; confirmation remains sealed.

The bundle preserves the analysis/record, actual rows, resource cells, preflight, source snapshot,
all 64 exact SSP2E/L/F/S cell streams, and all 16 SSPL1 inputs. `streams.sha256` binds the portable
stream inventory. Core metadata SHA-256 values:

- analysis `07cb37beb4bd9146bc10f1440c7672a177248a745892f66263477bed7f6043d2`;
- analysis record `0a0f354a563e78cb035995781f1ba360e9a8781099540cf82e6d063450366b57`;
- actual rows `19f3f1ad0f336c9ea59c6a89893fb93ac65aeace9ca8662e0750cf4efb526ec8`;
- benchmark cells `615e92b63ca1154d0d9e84742e0e1dfd584b9aa55c143d67b4f704094ef660ae`;
- source snapshot `fac4ca0978891b3cd16d477ffc04d12dc120168478192293d39af635fca7eb50`.

COMP-010 is lifecycle/provenance proof for this decision; it does not strengthen the compression
claim.
