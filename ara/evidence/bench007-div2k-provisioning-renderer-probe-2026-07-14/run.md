# BENCH-007 DIV2K provisioning and renderer feasibility probe

Date: 2026-07-14

## Dataset provenance

The canonical train-HR archive was downloaded from the official ETH Zurich DIV2K endpoint:

```text
https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
Content-Length: 3530603713
Last-Modified: Tue, 14 Feb 2017 14:27:11 GMT
ETag: "d270bcc1-5487e5c18d636"
SHA-256: 9d0b9c463f6e35b6c62cc6a930ee2224f670b34c1df841a57670f9acf0f6c335
```

`unzip -tq` reported no errors. Only the four preregistered Stage-0b calibration images and eight
preregistered Stage-1 pilot images were extracted to `../../../tests/test_images/DIV2K_train_HR/`. The
downloaded archive copy was then removed to retain space for complete SSPL1 candidate streams and
fitted fields. The source PNG and decoded contiguous RGB-byte identities are:

| ID | Dimensions | Pixels | Source PNG SHA-256 | Decoded RGB SHA-256 |
|---|---:|---:|---|---|
| 0001 | 2040x1404 | 2,864,160 | `cdb20d7a462744c269d8e197f735c7bc42e7cda367a940a9b7bc27803b1c8619` | `d8cc6f760ad5c53dca54c73707c68eac49d3e54c8a190643c04182018b973993` |
| 0002 | 2040x1848 | 3,769,920 | `82325cea74c2cd4681f69a10e36ba15c896d99ec47dc2c687ef07f7497781e09` | `8ec51fc4a5a0bd98e2102706a78256af5d0995cd8cf8fd8bf3b62dc5a2d57a82` |
| 0115 | 2040x1536 | 3,133,440 | `b08214ed8a205d5ff148eb14541de6117f282350bc3e4fc46d2efa8c848073e1` | `38936da42b394003ef6a4317d8e9e41843bdf944c3e4d414aad73bb4d927d215` |
| 0229 | 2040x1296 | 2,643,840 | `e985cdadc0861ae47a76ae66a46290b7aa322b4d2596727634b144cb205c2d18` | `a204d5fc3fc7dc7636e73372f4e334803377a7dd30d0f1f6e14fbdc51c7f5dce` |
| 0268 | 2040x1356 | 2,766,240 | `455a05afcc60e0638259bb6dd98018606786cd73ee7118049cff94b48b5d4e7b` | `82e9c5e25a072485b42099c61ffbee6b184ac172e7c3938c73cfd2f1fb808444` |
| 0343 | 2040x1200 | 2,448,000 | `f70f775deb82a5744fae0640b5b095e35374f7228893dead5750a4b9d7ef8781` | `bd21cf872e1d421bf7c95242363b853bec4f328c65eb5a64963028b0c01c72b6` |
| 0457 | 1164x2040 | 2,374,560 | `565bb5b65c50abd4b0715b9318851de400cae1475db9c44a138a3bae275d2a05` | `f24b5184088c2b17153ed2b41e41da30e1d7bf3b894a67cce1ed957edd2fb17d` |
| 0534 | 2040x1224 | 2,496,960 | `c605f2a1092cafc85280d618eb55344c58830313dc75b0469a8f7321f11aa4d3` | `008ab267dbf55d5dceec5db6144c60ded6e70ad6cf143d0d03dbbf6928c48a8d` |
| 0571 | 2040x1356 | 2,766,240 | `6de58e0706300b3496f538dca3b80d478062f4c4396990b3b5e6479300ed71ef` | `0dd243b67d127ddb11e3288cde33d5a5fb82209fd5fd02c5fca0d910b2cf4f2a` |
| 0685 | 2040x1080 | 2,203,200 | `c42e9a8e92f57ed8ebff3ba247c7578aa85b59785021123f673c56d895e63364` | `dd8855a31a7a799ee527171e737a578acbe8f807d8f58e393bc9a6ae00f9ae42` |
| 0799 | 2040x1356 | 2,766,240 | `ad42d7e2fe2ee15461e6999e7673a1f96b1be791b4be8c01baca26812f5667db` | `baeac416bf81db91362422d890d7212ae32d654efde7540b44066f81423b6424` |
| 0800 | 2040x1332 | 2,717,280 | `eb6df5bfeacd04334062b6103f6ee8f33af1abd3e1375a7f2c2a4831fa701221` | `e7cce986c550febae71660b56d4f52c0d48cfac8f346016b972916a4ee526433` |

No image was resized, cropped, reoriented, or re-encoded.

## Pre-freeze renderer probe

The preregistration requires the normalized weighted-sum renderer equation. On clean commit
`50a2351`, image `0002` was initialized once with `quadtree_wse`, seed 0, and 7,721 Gaussians
(2,048 Gaussians per megapixel), then reloaded into two identical two-iteration fits. Both ran on
the NVIDIA GeForce RTX 4090 at native 1848x2040 resolution.

| Implementation | Two-step wall time | Peak allocated VRAM | Final PSNR |
|---|---:|---:|---:|
| PyTorch normalized reference | 6.2790 s | 9,815.5 MiB | 21.270336 dB |
| Owned exact CUDA normalized | 0.3795 s | 949.1 MiB | 21.270336 dB |

Exact CUDA was 16.5x faster and used 10.3x less allocated VRAM in this probe. The identical scalar
PSNR is accompanied by the existing forward/backward parity matrix over normalization/additivity,
opacity, antialiasing dilation, and support fade. The focused parity plus BENCH-007 tests passed
28/28, and the full suite passed 466 tests with one CUDA build warning.

## Protocol decision

This feasibility check occurred before Stage-0b calibration and before the Stage-1 manifest was
frozen. BENCH-007 now freezes two separate facts:

1. the scientific equation is `normalized_weighted_sum`; and
2. the implementation is either `pytorch_reference` or `owned_exact_cuda`.

Stage-0b and Stage-1 use the owned exact-CUDA implementation, the same equation for every arm, and
the existing `1e-6` cold-field parity requirement. This is an implementation-level feasibility
choice made without viewing Stage-1 outcomes; it is not an outcome-contingent rescue or a change to
the comparison class.
