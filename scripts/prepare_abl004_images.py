#!/usr/bin/env python
"""Prepare the ABL-004 image manifest without committing downloaded data.

Downloads Kodak-24 into an ignored results directory and combines it with the repo's pinned
COCO fixture images. The ablation runner accepts the generated images.txt directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

KODAK_BASE_URL = "https://r0k.us/graphics/kodak/kodak"
DEFAULT_OUTDIR = Path("results/datasets/abl004")
DEFAULT_COCO_DIR = Path("tests/test_images")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def kodak_items() -> list[tuple[str, str]]:
    return [
        (f"kodim{i:02d}.png", f"{KODAK_BASE_URL}/kodim{i:02d}.png")
        for i in range(1, 25)
    ]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "structsplat-abl004/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)


def collect_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def image_record(split: str, path: Path, url: str | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "split": split,
        "name": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if url is not None:
        rec["url"] = url
    return rec


def build_manifest(kodak_files: list[Path], coco_files: list[Path]) -> dict[str, Any]:
    kodak_url_by_name = dict(kodak_items())
    images = [
        image_record("kodak24", p, kodak_url_by_name.get(p.name))
        for p in sorted(kodak_files)
    ]
    images.extend(image_record("coco_pinned", p) for p in sorted(coco_files))
    return {
        "description": "ABL-004 image set: Kodak-24 plus pinned COCO fixture images.",
        "kodak_source": KODAK_BASE_URL,
        "coco_source": str(DEFAULT_COCO_DIR),
        "images": images,
    }


def write_outputs(outdir: Path, manifest: dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [rec["path"] for rec in manifest["images"]]
    (outdir / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(outdir: Path, coco_dir: Path, skip_download: bool = False) -> dict[str, Any]:
    kodak_dir = outdir / "kodak24"
    if not skip_download:
        for name, url in kodak_items():
            download_file(url, kodak_dir / name)
    kodak_files = collect_images(kodak_dir)
    coco_files = collect_images(coco_dir)
    if len(kodak_files) != 24:
        raise SystemExit(f"expected 24 Kodak images in {kodak_dir}, found {len(kodak_files)}")
    if not coco_files:
        raise SystemExit(f"no pinned COCO images found in {coco_dir}")
    manifest = build_manifest(kodak_files, coco_files)
    write_outputs(outdir, manifest)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare ABL-004 Kodak+COCO image manifest")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--coco-dir", type=Path, default=DEFAULT_COCO_DIR)
    p.add_argument("--skip-download", action="store_true",
                   help="only validate existing Kodak files and rewrite manifest/images.txt")
    a = p.parse_args()
    manifest = prepare(a.outdir, a.coco_dir, skip_download=a.skip_download)
    counts: dict[str, int] = {}
    for rec in manifest["images"]:
        counts[rec["split"]] = counts.get(rec["split"], 0) + 1
    print(f"wrote {a.outdir / 'manifest.json'} and images.txt with counts {counts}")


if __name__ == "__main__":
    main()
