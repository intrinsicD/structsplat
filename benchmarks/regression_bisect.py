"""ABL-003 four-image regression bisect runner.

Runs the StructSplat arm of the historical matched COCO comparison at PR #2 commits in detached
worktrees. The script keeps downloaded images and temporary worktrees under ignored paths, then
writes compact evidence tables under ``ara/evidence``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from urllib.request import urlretrieve

from benchmarks.common import write_csv


DEFAULT_IMAGES = [
    "COCO_train2014_000000000009.jpg",
    "COCO_train2014_000000000025.jpg",
    "COCO_train2014_000000000030.jpg",
    "COCO_train2014_000000000034.jpg",
]

COMMITS = [
    ("merge_base", "f49aa18", "PR #2 merge base"),
    ("stage_influence", "71fad3e", "stage-influence ablation mode and stage variants"),
    ("correctness_fixes", "ef730a9", "renderer/fit/init correctness and validity fixes"),
    ("two_sided_fix", "a455e98", "two_sided flank color side-selection fix"),
]

CHILD_RUNNER = r"""
import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import fit
from structsplat.init import build_field


def _load_image(path, max_side):
    img = Image.open(path).convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1.0:
        img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _make(cls, **kwargs):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in kwargs.items() if k in names})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--iters", type=int, required=True)
    p.add_argument("--max-side", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", required=True)
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested cuda but torch.cuda.is_available() is false")

    rows = []
    fit_cfg = _make(
        FitConfig,
        iters=args.iters,
        render_chunk=384,
        log_every=max(1, args.iters // 4),
        target_psnrs=[25.0, 28.0, 30.0],
        compute_lpips=False,
    )
    scfg = StructureTensorConfig()
    for image_path in args.images:
        img = _load_image(Path(image_path), args.max_side)
        target = torch.as_tensor(img, dtype=torch.float32, device=device)
        torch.manual_seed(args.seed)
        start = time.time()
        field = build_field(
            img,
            _make(InitConfig, strategy="aniso_flanking", num_gaussians=args.budget, seed=args.seed),
            scfg,
            device=device,
        )
        out = fit(field, target, fit_cfg, verbose=False)
        rows.append({
            "image": Path(image_path).stem,
            "source_path": str(image_path),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "budget": args.budget,
            "iters": args.iters,
            "max_side": args.max_side,
            "seed": args.seed,
            "device": device,
            "strategy": "aniso_flanking",
            "psnr": round(float(out["psnr"]), 4),
            "ssim": round(float(out["ssim"]), 5),
            "ms_ssim": round(float(out["ms_ssim"]), 5),
            "n_gaussians": int(out["n_gaussians"]),
            "fit_seconds": round(float(out.get("fit_seconds", time.time() - start)), 4),
            "total_seconds": round(float(time.time() - start), 4),
        })
        print(rows[-1], flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Run ABL-003 four-image regression bisect")
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--dataset-dir", type=Path, default=Path("results/abl003_coco_train2014"))
    p.add_argument("--outdir", type=Path,
                   default=Path("ara/evidence/abl003-regression-bisect-2026-07-03"))
    p.add_argument("--worktree-dir", type=Path, default=Path("results/abl003_worktrees"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--keep-worktrees", action="store_true",
                   help="leave detached worktrees under --worktree-dir for debugging")
    p.add_argument("--device", default="cpu")
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--max-side", type=int, default=160)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    repo = args.repo.resolve()
    dataset_dir = (repo / args.dataset_dir).resolve()
    outdir = (repo / args.outdir).resolve()
    worktree_dir = (repo / args.worktree_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    worktree_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        _download_images(dataset_dir)
    image_paths = [dataset_dir / name for name in DEFAULT_IMAGES]
    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing COCO images: {missing}; rerun with --download")

    resolved_commits = []
    for label, rev, note in COMMITS:
        sha = _git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()
        resolved_commits.append((label, sha, note))

    all_rows: list[dict] = []
    commit_summaries: list[dict] = []
    for idx, (label, sha, note) in enumerate(resolved_commits):
        wt = worktree_dir / f"{idx:02d}_{label}_{sha[:7]}"
        if wt.exists():
            shutil.rmtree(wt)
        _git(repo, "worktree", "add", "--detach", str(wt), sha)
        run_out = outdir / f"{idx:02d}_{label}_{sha[:7]}.json"
        log_out = outdir / f"{idx:02d}_{label}_{sha[:7]}.log"
        cmd = [
            sys.executable, "-c", CHILD_RUNNER,
            "--images", *[str(p) for p in image_paths],
            "--out", str(run_out),
            "--budget", str(args.budget),
            "--iters", str(args.iters),
            "--max-side", str(args.max_side),
            "--seed", str(args.seed),
            "--device", args.device,
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(wt / "src"), str(wt), env.get("PYTHONPATH", "")])
        proc = subprocess.run(cmd, cwd=wt, env=env, text=True, capture_output=True, check=False)
        log_out.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise SystemExit(f"{label} {sha[:7]} failed; see {log_out}")
        if not args.keep_worktrees:
            _git(repo, "worktree", "remove", "--force", str(wt))
        rows = json.loads(run_out.read_text(encoding="utf-8"))
        psnrs = [float(r["psnr"]) for r in rows]
        summary = {
            "order": idx,
            "label": label,
            "commit": sha,
            "short": sha[:7],
            "note": note,
            "mean_psnr": round(mean(psnrs), 4),
            "std_psnr": round(pstdev(psnrs) if len(psnrs) > 1 else 0.0, 4),
            "runs": len(rows),
        }
        commit_summaries.append(summary)
        for row in rows:
            all_rows.append({"order": idx, "commit_label": label, "commit": sha, **row})

    _write_csv(outdir / "metrics.csv", all_rows)
    _write_csv(outdir / "commit_summary.csv", commit_summaries)
    (outdir / "metrics.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    (outdir / "config.json").write_text(json.dumps({
        "command": " ".join(sys.argv),
        "repo": str(repo),
        "dataset_dir": str(dataset_dir),
        "image_names": DEFAULT_IMAGES,
        "commits": resolved_commits,
        "budget": args.budget,
        "iters": args.iters,
        "max_side": args.max_side,
        "seed": args.seed,
        "device": args.device,
        "runner": "benchmarks.regression_bisect",
    }, indent=2), encoding="utf-8")
    _write_summary(outdir / "summary.md", commit_summaries, all_rows)
    print(f"wrote ABL-003 evidence to {outdir}")


def _download_images(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name in DEFAULT_IMAGES:
        dst = dataset_dir / name
        if dst.exists():
            continue
        urlretrieve(f"http://images.cocodataset.org/train2014/{name}", dst)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    write_csv(path, rows, sort_fieldnames=True)


def _write_summary(path: Path, commits: list[dict], rows: list[dict]) -> None:
    lines = [
        "# ABL-003 Regression Bisect",
        "",
        "Four-image COCO train2014 rerun of the StructSplat `aniso_flanking` arm only.",
        "Each row uses max-side 160, budget 512, seed 0, 80 fit iterations, and CPU unless",
        "the config says otherwise.",
        "",
        "## Per-Commit Mean",
        "",
        "| Order | Commit | Label | Mean PSNR | Std | Delta vs previous | Note |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    prev = None
    for item in commits:
        delta = "" if prev is None else f"{item['mean_psnr'] - prev:.4f}"
        lines.append(
            f"| {item['order']} | `{item['short']}` | {item['label']} | "
            f"{item['mean_psnr']:.4f} | {item['std_psnr']:.4f} | {delta or '-'} | "
            f"{item['note']} |"
        )
        prev = item["mean_psnr"]

    lines += [
        "",
        "## Per-Image PSNR",
        "",
        "| Image | " + " | ".join(f"`{c['short']}`" for c in commits) + " |",
        "|---|" + "---:|" * len(commits),
    ]
    by_image_commit = {(r["image"], r["commit"]): r for r in rows}
    for image in DEFAULT_IMAGES:
        stem = Path(image).stem
        cells = []
        for commit in commits:
            row = by_image_commit[(stem, commit["commit"])]
            cells.append(f"{row['psnr']:.4f}")
        lines.append(f"| `{stem}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Mechanism Notes",
        "",
        "- Compare `71fad3e..ef730a9` first: that diff changes blue-noise spacing-scale semantics",
        "  from exclusion radius to cell-side spacing (`sqrt(pi)` larger), clips reference-render",
        "  supports to image bounds, prevents final-iteration restructure, and fixes opacity padding.",
        "- Compare `ef730a9..a455e98` second: that diff is limited to the `two_sided` color-side",
        "  correction; this run uses the historical StructSplat arm's default `color_mode=bilinear`,",
        "  so it should not move this exact baseline unless hidden defaults changed.",
        "",
        "Raw rows: `metrics.json` / `metrics.csv`; command and environment: `config.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
