# FIT-019 opacity-split gauge-equivalence audit

Commutation confirmed: **True**.
Recovery-utility guard survives: **False**.

| Arm | Immediate PSNR | Post-20 PSNR | Post-100 PSNR | Post-100 AUC | Unique groups | Total-100 s |
|---|---:|---:|---:|---:|---:|---:|
| `support_canonical` | 24.4108 | 29.6021 | 35.7529 | 32.4442 | 8.00 | 0.9180 |
| `support_gauge_row` | 24.7670 | 29.1184 | 36.8959 | 32.7965 | 5.50 | 0.8971 |
| `responsibility_alpha0.7_canonical` | 24.3643 | 29.6832 | 35.8554 | 32.4505 | 8.00 | 0.8863 |
| `responsibility_alpha0.7_gauge_row` | 24.2840 | 29.0989 | 36.9735 | 32.8465 | 6.12 | 0.8826 |
| `responsibility_alpha1_canonical` | 24.3191 | 29.5356 | 36.1731 | 32.4894 | 8.00 | 0.9021 |
| `responsibility_alpha1_gauge_row` | 24.4238 | 29.3245 | 36.7738 | 32.6394 | 5.38 | 0.8922 |
| `quotient_alpha1_canonical` | 24.3191 | 29.5356 | 36.1731 | 32.4894 | 8.00 | 0.9024 |
| `quotient_alpha1_gauge` | 24.3191 | 29.5356 | 36.1731 | 32.4894 | 8.00 | 0.9064 |

Raw alpha-1 action changed on 8/8 target clusters.

Grouped alpha-1 versus gauge-row alpha-1: post-20 +0.2111 dB, post-100 -0.6007 dB, post-20 wins 5/8 targets.

Grouped alpha-1 versus canonical support: post-20 -0.0665 dB, post-100 +0.4202 dB; total-100 overhead -1.3%.

This procedural benchmark is a mechanism guard. A commutation-only pass is a diagnostic,
not authorization for production lineage metadata, natural-image claims, or defaults.
