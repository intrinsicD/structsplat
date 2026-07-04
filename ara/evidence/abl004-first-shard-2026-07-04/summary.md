# ABL-004 first full-protocol shard

This is not the completed ABL-001 sweep. It records the first bounded ABL-004 shard used to
validate the resumable full-run path and estimate local compute cost.

Command:

```bash
MAX_NEW_CELLS=1 OUTDIR=results/abl004_ablation_full MAX_SIDE=768 DEVICE=cuda \
  PYTHONPATH=src:. scripts/run_abl004_full_ablation.sh
```

Result:

- Dataset prep succeeded: 24 Kodak images plus 4 pinned COCO fixtures in
  `results/datasets/abl004/images.txt`.
- One full-protocol cell completed and was appended to `results/abl004_ablation_full/ablation.jsonl`.
- Cell: `kodim01`, strategy `random`, budget 2000, seed 0, max-side 768, 1500 iterations.
- Metrics: PSNR 22.2870 dB, SSIM 0.64317, MS-SSIM 0.85389; target 35 dB not reached.
- Timing: init 0.0091 s, fit 780.3784 s on CUDA (`NVIDIA GeForce RTX 3050`).

Interpretation:

The official ABL-004 matrix is now launchable in resumable shards, but a single low-budget
full-resolution cell takes about 13 minutes on the available RTX 3050. The complete
28-image x 4-budget x 3-seed x 11-arm sweep is therefore a multi-day to multi-week local
job unless run on faster hardware, reduced resolution/iterations, or a distributed queue.
