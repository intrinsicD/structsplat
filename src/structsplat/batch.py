"""Multi-process batch encoding across images (PORT-005).

Per-image fits are independent, so production encode *throughput* (images/hour) scales by
running worker processes in parallel — round-robin across the requested devices — instead of
optimizing single-fit latency. Workers pull image paths from a shared queue (dynamic load
balancing for mixed image sizes), fit with exactly the `fit`-CLI option surface via
:func:`structsplat.cli.build_fit_configs`, and stream one JSON metrics row per image back to the
parent, which appends them to ``<outdir>/metrics.jsonl``.

Reproducibility: every row logs the JSON-safe option surface, the seed, and the device. The
configured ``--seed`` seeds each image's init identically (per-image variation comes from the
image content, matching the ablation harness's explicit-seed practice). On CUDA, run-to-run
bitwise equality is still bounded by atomic accumulation (ADR-0011); rows therefore log device
and renderer. Resume: images whose output ``.npz`` already exists are skipped unless ``--force``.

Mask-contained fitting (CORE-010) is per-image-mask work and stays with the single-image `fit`
command; batch-fit rejects mask options rather than guessing a pairing convention.
"""
from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import queue as _queue
import time

_IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
_SENTINEL = None


def expand_images(items: list[str]) -> list[str]:
    """Files, directories (common image extensions), and glob patterns -> sorted file list."""
    files: list[str] = []
    for item in items:
        if os.path.isdir(item):
            for pattern in _IMAGE_PATTERNS:
                files += glob.glob(os.path.join(item, pattern))
        elif any(ch in item for ch in "*?["):
            files += glob.glob(item)
        else:
            files.append(item)
    return sorted(dict.fromkeys(files))


def unique_bases(paths: list[str]) -> list[str]:
    """Deterministic collision-free output basenames (same-named files from different dirs)."""
    seen: dict[str, int] = {}
    bases = []
    for path in paths:
        base = os.path.splitext(os.path.basename(path))[0]
        count = seen.get(base, 0)
        seen[base] = count + 1
        bases.append(base if count == 0 else f"{base}_{count}")
    return bases


def json_safe_args(args) -> dict:
    """The reproducibility surface of the parsed options: everything JSON can carry."""
    safe = {}
    for key, value in sorted(vars(args).items()):
        if key == "func":
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            safe[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (str, int, float, bool, type(None))) for v in value
        ):
            safe[key] = list(value)
    return safe


def resolve_devices(args) -> list[str]:
    if getattr(args, "devices", None):
        return [d.strip() for d in str(args.devices).split(",") if d.strip()]
    import torch

    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def run_single_fit(args, image_path: str, base: str, device: str) -> dict:
    """One image through init + fit on `device`, saving the same artifacts `fit` saves."""
    import torch

    from . import init as _init
    from .cli import build_fit_configs, load_image, save_image
    from .config import PyramidConfig
    from .fit import fit
    from .pyramid import fit_pyramid

    t_total = time.perf_counter()
    img = load_image(image_path)
    target = torch.as_tensor(img, device=device)
    icfg, scfg, fcfg = build_fit_configs(args)

    init_seconds = None
    t_fit = time.perf_counter()
    if args.pyramid:
        per_level = args.iters_per_level or max(1, args.iters // max(1, args.pyramid_levels))
        pcfg = PyramidConfig(levels=args.pyramid_levels,
                             level_fractions=args.level_fractions,
                             iters_per_level=per_level,
                             level_iters=args.level_iters)
        out = fit_pyramid(img, target, icfg, fcfg, pcfg, scfg)
    else:
        t_init = time.perf_counter()
        field = _init.build_field(img, icfg, scfg, device=device)
        init_seconds = time.perf_counter() - t_init
        t_fit = time.perf_counter()
        out = fit(field, target, fcfg, verbose=False)
    fit_seconds = time.perf_counter() - t_fit

    os.makedirs(args.outdir, exist_ok=True)
    npz_path = os.path.join(args.outdir, f"{base}_{args.strategy}.npz")
    png_path = os.path.join(args.outdir, f"{base}_{args.strategy}.png")
    save_image(png_path, out["render"].detach().cpu().numpy())
    out["field"].save(npz_path)

    row = {
        "image": image_path,
        "base": base,
        "device": device,
        "seed": int(args.seed),
        "height": int(img.shape[0]),
        "width": int(img.shape[1]),
        "n_gaussians": int(out["n_gaussians"]),
        "psnr": float(out["psnr"]),
        "ssim": float(out["ssim"]),
        "ms_ssim": float(out["ms_ssim"]),
        "lpips": None if out.get("lpips") is None else float(out["lpips"]),
        "iters_to_target": out["iters_to_target"],
        "init_seconds": init_seconds,
        "fit_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - t_total,
        "out_npz": npz_path,
        "out_png": png_path,
    }
    return row


def _worker_main(work_queue, result_queue, args, device: str) -> None:
    import torch

    if args.torch_threads:
        torch.set_num_threads(int(args.torch_threads))
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    while True:
        task = work_queue.get()
        if task is _SENTINEL:
            return
        image_path, base = task
        try:
            row = run_single_fit(args, image_path, base, device)
        except Exception as exc:  # per-image isolation: one bad image must not kill the farm
            row = {"image": image_path, "base": base, "device": device,
                   "seed": int(args.seed), "error": f"{type(exc).__name__}: {exc}"}
        result_queue.put(row)


def run_batch(args) -> list[dict]:
    if getattr(args, "mask", None) or getattr(args, "mask_contain", False) or \
            getattr(args, "mask_coverage_weight", 0.0) or \
            getattr(args, "loss_weighting", "none") == "mask":
        raise SystemExit("batch-fit does not support mask options; use `fit` per image")

    images = expand_images(args.images)
    if not images:
        raise SystemExit(f"no images found under {args.images!r}")
    bases = unique_bases(images)

    pending: list[tuple[str, str]] = []
    skipped = 0
    for image_path, base in zip(images, bases):
        npz_path = os.path.join(args.outdir, f"{base}_{args.strategy}.npz")
        if not args.force and os.path.exists(npz_path):
            skipped += 1
            continue
        pending.append((image_path, base))

    devices = resolve_devices(args)
    workers = args.workers or (len(devices) if devices[0].startswith("cuda") else 1)
    workers = max(1, min(workers, max(1, len(pending))))
    print(f"batch-fit: {len(pending)} to fit, {skipped} skipped (resume), "
          f"{workers} workers over {devices}")
    if not pending:
        return []

    os.makedirs(args.outdir, exist_ok=True)
    config = json_safe_args(args)
    rows: list[dict] = []
    metrics_path = os.path.join(args.outdir, "metrics.jsonl")

    ctx = mp.get_context("spawn")
    work_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for task in pending:
        work_queue.put(task)
    procs = []
    for w in range(workers):
        work_queue.put(_SENTINEL)
        proc = ctx.Process(target=_worker_main,
                           args=(work_queue, result_queue, args, devices[w % len(devices)]),
                           daemon=True)
        proc.start()
        procs.append(proc)

    with open(metrics_path, "a", encoding="utf-8") as sink:
        while len(rows) < len(pending):
            try:
                row = result_queue.get(timeout=30)
            except _queue.Empty:
                if not any(proc.is_alive() for proc in procs):
                    # A hard worker death (OOM kill, segfault) loses its in-flight image; the
                    # metrics rows already on disk stay valid and a resume rerun picks up the
                    # remainder, so fail loudly instead of hanging the farm.
                    raise SystemExit(
                        f"batch-fit: workers exited early with {len(pending) - len(rows)} "
                        f"images unaccounted; rerun to resume from {metrics_path}"
                    )
                continue
            row["config"] = config
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            rows.append(row)
            if row.get("error"):
                print(f"  {row['base']}: ERROR {row['error']}")
            else:
                print(f"  {row['base']}: PSNR {row['psnr']:.2f} | "
                      f"{row['fit_seconds']:.1f}s fit on {row['device']}")
    for proc in procs:
        proc.join()

    done = [r for r in rows if not r.get("error")]
    failed = [r for r in rows if r.get("error")]
    if done:
        mean_psnr = sum(r["psnr"] for r in done) / len(done)
        total_fit = sum(r["fit_seconds"] for r in done)
        print(f"batch-fit: {len(done)} ok (mean PSNR {mean_psnr:.2f}, "
              f"{total_fit:.1f}s summed fit time), {len(failed)} failed -> {metrics_path}")
    else:
        print(f"batch-fit: 0 ok, {len(failed)} failed -> {metrics_path}")
    return rows
