# BENCH-019 downstream-protocol preflight — 2026-08-03

## Classification

Read-only diagnostic inventory. This is neither a prospective protocol approval nor a downstream
result. No field was loaded, converted, fitted, stopped, or changed. The active mask-contained
producer was left running.

## Question

Can BENCH-019 freeze the required general-surrogate protocol from the currently available Janelle
stage artifacts before downstream outcomes are accessed?

## Read-only checks

```bash
find /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric -maxdepth 1 \
  -mindepth 1 -type d -printf '%f\n' | sort
find .../frame_00008/PROVIDER -maxdepth 1 -name '*.rtgsv' | wc -l
find .../frame_00009 -maxdepth 1 -mindepth 1 -printf '%f\t%y\n' | sort
ps -eo pid,etimes,args | rg 'structsplat_mask_contained'
jq . /home/alex/Documents/realtime-gs/experiments/tasks/\
20260801_paper_three_provider_fullres_stage_frame00008.json
jq . /home/alex/Documents/realtime-gs/experiments/data/\
stage_frame00008_three_provider_fullres.json
sha256sum <the two complete provider manifests and the two realtime-gs protocol inputs>
```

The exact snapshot is in [`inventory.json`](inventory.json).

## Findings

- The capture root exposes two frames, but they are part of one capture group.
- Frame 00008 had complete GaussianImage and StructSplat-no-boundary manifests. The
  mask-contained producer had seven `.rtgsv` files, no terminal manifest, and an active C0014
  worker at the snapshot time.
- Frame 00009 had the legacy `gaussians2d`, RGB, and mask directories, not the matched three
  full-resolution 11k provider directories required for a paired family analysis.
- The corresponding realtime-gs task was still `draft`; its data seal listed zero datasets, no
  distinct prospective review existed, and no downstream result had run.
- Both repositories were dirty, which the implemented BENCH-019 `prepare-review` command rejects
  for formal work by design.

## Disposition

Do not weaken the protocol. Finish and seal the matched frame-00008 producer set, produce the same
families for frame 00009, and identify/acquire two additional independent capture groups before a
general Stage-1 surrogate claim. The user's eventual same-frame comparison may still run under an
explicit `workload_specific` manifest, but it cannot promote a general surrogate or production
default by itself.
