# FIT-018 responsibility split guard

Every arm starts from the same fitted field and performs the same moment-preserving
split. Recovery horizons are independently replayed from that post-split field.

| Site score | Immediate PSNR | Post-20 PSNR | Post-100 PSNR | Score s | Total-100 s |
|---|---:|---:|---:|---:|---:|
| `residual` | 20.3105 | 21.5835 | 23.0925 | 0.000091 | 1.001885 |
| `support` | 20.3093 | 21.6255 | 23.2022 | 0.003158 | 0.987286 |
| `responsibility_alpha1` | 20.3210 | 21.6274 | 23.2444 | 0.005838 | 0.996310 |
| `responsibility_alpha0.7` | 20.2470 | 21.6057 | 23.1611 | 0.005745 | 1.009991 |

Comparator selected before donor deltas: `support` (higher mean post-20 PSNR of residual/support).

Donor deltas: post-20 -0.0198 dB; post-100 -0.0411 dB; positive post-20 pairs 4/8; total-100 overhead +2.3%.

Preregistered guard survives: **False**.

This reused-fixture mechanism smoke does not support default, SOTA, or compression claims.
