# ABL-003 Regression Bisect

Four-image COCO train2014 rerun of the StructSplat `aniso_flanking` arm only.
Each row uses max-side 160, budget 512, seed 0, 80 fit iterations, and CPU unless
the config says otherwise.

## Per-Commit Mean

| Order | Commit | Label | Mean PSNR | Std | Delta vs previous | Note |
|---:|---|---|---:|---:|---:|---|
| 0 | `f49aa18` | merge_base | 24.8202 | 3.1519 | - | PR #2 merge base |
| 1 | `71fad3e` | stage_influence | 24.8202 | 3.1519 | 0.0000 | stage-influence ablation mode and stage variants |
| 2 | `ef730a9` | correctness_fixes | 24.0510 | 2.9524 | -0.7692 | renderer/fit/init correctness and validity fixes |
| 3 | `a455e98` | two_sided_fix | 24.0510 | 2.9524 | 0.0000 | two_sided flank color side-selection fix |

## Per-Image PSNR

| Image | `f49aa18` | `71fad3e` | `ef730a9` | `a455e98` |
|---|---:|---:|---:|---:|
| `COCO_train2014_000000000009` | 24.5217 | 24.5217 | 23.6543 | 23.6543 |
| `COCO_train2014_000000000025` | 23.4105 | 23.4105 | 22.8291 | 22.8291 |
| `COCO_train2014_000000000030` | 29.9290 | 29.9290 | 28.8544 | 28.8544 |
| `COCO_train2014_000000000034` | 21.4198 | 21.4198 | 20.8662 | 20.8662 |

## Mechanism Notes

- Compare `71fad3e..ef730a9` first: that diff changes blue-noise spacing-scale semantics
  from exclusion radius to cell-side spacing (`sqrt(pi)` larger), clips reference-render
  supports to image bounds, prevents final-iteration restructure, and fixes opacity padding.
- Compare `ef730a9..a455e98` second: that diff is limited to the `two_sided` color-side
  correction; this run uses the historical StructSplat arm's default `color_mode=bilinear`,
  so it should not move this exact baseline unless hidden defaults changed.

Raw rows: `metrics.json` / `metrics.csv`; command and environment: `config.json`.
