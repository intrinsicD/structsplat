# ABL-006 Successive-Halving Harness Evidence

Date: 2026-07-07

This is a harness/protocol artifact, not the final ABL-006 confirmation run. It verifies that
`benchmarks.abl004_confirmation halving-plan` emits the frozen elimination-rule JSON, staged cell
manifest, run groups, and elimination trail.

Command:

```bash
python -m benchmarks.abl004_confirmation halving-plan \
  --outdir results/abl006_halving_plan_2026_07_07 \
  --target-psnrs 28 30 32
```

Result:

- `scripts/prepare_abl004_images.py` prepared the intended image set:
  Kodak-24 plus 4 pinned COCO fixtures.
- The first-stage default plan contains 336 cells:
  28 images x budget 2000 x seeds {0,1} x 6 arms.
- The committed decisions file has no survivor decisions yet, so only stage 1 is planned.
- The final ABL-006 run still needs the staged GPU execution and survivor decisions recorded
  after each stage.

Focused validation:

```bash
python -m py_compile benchmarks/abl004_confirmation.py
python -m pytest tests/test_abl004_confirmation.py
```

Both passed.
