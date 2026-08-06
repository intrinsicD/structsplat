# HIER-005 pixel-contraction implementation diagnostic

## Scope

This note preserves a dirty-worktree, single-source implementation smoke for the default-off
HIER-005 reference. It is not a preregistered experiment, maintained report bundle, independent
review, semantic selection, codec result, incumbent comparison, or default claim. The two temporary
HTML reports were not copied into the repository; their metric-file hashes are recorded only as
diagnostic receipts.

The implementation, driver, tests, and synchronized architecture documents are bound by source-set
SHA-256 `e8ab01da5711828ce3e1bdb19a1ddfcb398fc5c4aaf8aa69ab03d2d634391a1f`. HIER-005 remains
`in-review` and CORE-013/BENCH-020 remain unresolved authorities.

## Source

- Local path: `results/native_image_gs_harness_smoke_v2/targets/COCO_train2014_000000000009_35cdfe8259ac_s64.png`
- Decoded shape: `48x64` RGB (`3,072` pixels)
- Supplied-file bytes: `7,831`
- SHA-256: `87f3de9b337c00f942c8b5e693edba84d9b2ba67bade1548a8b4149ec43fafc5`
- Provenance boundary: pre-existing local `results/` diagnostic source, not a newly frozen held-out
  fixture and not a general dataset.

## Commands

```bash
python scripts/experiments/hier005_pixel_contraction.py \
  --images results/native_image_gs_harness_smoke_v2/targets/COCO_train2014_000000000009_35cdfe8259ac_s64.png \
  --out /tmp/structsplat-hier005-8y7k2U/report \
  --target-gaussians 1024 --device cpu --renderer additive \
  --proposal-batch-size 64 --merge-batch-size 16

python scripts/experiments/hier005_pixel_contraction.py \
  --images results/native_image_gs_harness_smoke_v2/targets/COCO_train2014_000000000009_35cdfe8259ac_s64.png \
  --out /tmp/structsplat-hier005-8y7k2U/report_pair_always \
  --target-gaussians 1024 --device cpu --renderer additive \
  --proposal-batch-size 64 --merge-batch-size 16 --pair-policy always
```

## Diagnostic rows

Both rows use direct signed coefficients, `leaf_scale_px=0.18`, zero support fade, the 3-sigma
rounded AABB support, a 32-byte uncoded row price, cold lossless Field V2 load, and the maintained
CPU additive renderer.

| pair policy | N | actions | PSNR dB | SSIM | MS-SSIM | contract s | cold decode s | render s | maintained max abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact_count` | 1,024 | 796 | 23.523126 | 0.890252 | 0.981815 | 1.982056 | 0.001145 | 0.003330 | 7.7486e-7 |
| `always` | 1,024 | 2,039 | 24.828353 | 0.908749 | 0.988585 | 2.882758 | 0.001138 | 0.003231 | 5.9605e-7 |

For both rows, estimated/raw payload is `32,768` bytes and deterministic lossless reference NPZ is
`35,456` bytes. Therefore supplied-source/estimated-payload ratio is `0.23898`: this uncoded field
is larger, not smaller, than the supplied PNG. The byte values are reference accounting only; they
do not substitute for COMP-013 complete coded bytes.

- `exact_count` field canonical hash:
  `4c435e57959942d461301f42e8c84d71e5d07bb770712366a323d3f0ca1b5d6e`
- `always` field canonical hash:
  `02139e5f26cd726f00fce5eed0b6e5959f716a91d2da37d3e68948d03029737f`
- Temporary `exact_count` metrics SHA-256:
  `400a027009e2f7eebd5a26de47891e113e80f48033157f97a0cbf8632fbf852d`
- Temporary `always` metrics SHA-256:
  `780fe28f44c3540008f96277ae70816140700c0fc229c363422c3e9d63d208eb`

The single-source delta shows that the exposed quality-first policy took more actions/time and
produced a higher-quality row on this diagnostic. It does not establish a general quality/speed
frontier or superiority to sparse initialization, current StructSplat, or another method.

## Verification

- Focused HIER-005 tests: `14 passed`.
- Field/render regression slice: `80 passed`.
- `./scripts/verify.sh`: `1,574 passed`, `4 skipped`, `514 deselected`; lint, docs sync, ARA,
  task policy, script layout, and agent workflow all passed.

## Required next evidence

A distinct numerical review must reproduce the bound implementation. After a reviewed semantic
selection, FIT-045 must compare this arm with fixed-N/global/regional/incumbent controls under
matched rows, work, and wall time. COMP-013/FIT-030 must replace estimated row price with complete
cold-decoded byte selection before any compression or rate-distortion claim.
