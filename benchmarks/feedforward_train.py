"""Train the tiny FF-001 feed-forward Gaussian predictor from exported teacher fields.

This is deliberately a minimal trainer: it proves that StructSplat can learn and load a model
checkpoint through `strategy=feedforward`. It is not the final amortized predictor architecture.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.common import load_image, run_config, write_config, write_csv, write_json


def _manifest_max_side(manifest: dict) -> int | None:
    resolved = manifest.get("config", {}).get("resolved", {})
    value = resolved.get("max_side")
    return None if value is None else int(value)


def _load_teacher_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", [])
    if not rows:
        raise ValueError(f"teacher manifest {manifest_path} has no rows")
    return manifest


def train_feedforward_predictor(
    manifest_path: str | Path,
    *,
    outdir: str | Path = "results/feedforward_train",
    image_size: int = 64,
    hidden: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | None = None,
    num_gaussians: int | None = None,
    max_scale_norm: float = 0.5,
) -> dict:
    """Train a tiny CNN predictor from a teacher manifest and save `predictor.pt`."""

    import torch

    from structsplat.gaussians import GaussianField
    from structsplat.predictor import (
        TinyGaussianPredictorNet,
        field_to_normalized_targets,
        image_batch_tensor,
        make_learned_checkpoint,
        prediction_loss,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = _load_teacher_manifest(manifest_path)
    rows = manifest["rows"]
    max_side = _manifest_max_side(manifest)
    if num_gaussians is None:
        num_gaussians = max(int(r.get("n_gaussians", r.get("budget", 0))) for r in rows)
    if int(num_gaussians) <= 0:
        raise ValueError(f"num_gaussians must be positive, got {num_gaussians}")

    images = []
    target_rows: dict[str, list[torch.Tensor]] = {
        "means": [],
        "scales": [],
        "rot_unit": [],
        "colors": [],
        "opacities": [],
    }
    for row in rows:
        img = load_image(row["source_path"], max_side)
        field = GaussianField.load(row["field_path"], device=device)
        images.append(img)
        targets = field_to_normalized_targets(
            field,
            img.shape[0],
            img.shape[1],
            int(num_gaussians),
            device=device,
            max_scale_norm=max_scale_norm,
        )
        for key, value in targets.items():
            target_rows[key].append(value)

    x = image_batch_tensor(images, int(image_size), device=device)
    targets_batched = {key: torch.stack(values, dim=0) for key, values in target_rows.items()}
    model = TinyGaussianPredictorNet(int(num_gaussians), hidden=int(hidden)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    history = []
    for epoch in range(1, int(epochs) + 1):
        opt.zero_grad(set_to_none=True)
        raw = model(x)
        loss = prediction_loss(raw, targets_batched, max_scale_norm=max_scale_norm)
        loss.backward()
        opt.step()
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})

    cfg = run_config({
        "manifest_path": str(manifest_path),
        "rows": len(rows),
        "image_size": int(image_size),
        "hidden": int(hidden),
        "epochs": int(epochs),
        "lr": float(lr),
        "seed": int(seed),
        "num_gaussians": int(num_gaussians),
        "max_scale_norm": float(max_scale_norm),
    }, device=device)
    write_config(str(out), cfg)
    checkpoint_path = out / "predictor.pt"
    torch.save(
        make_learned_checkpoint(
            model,
            image_size=int(image_size),
            max_scale_norm=float(max_scale_norm),
            train_config=cfg,
            loss_history=history,
        ),
        checkpoint_path,
    )
    write_json(out / "loss_history.json", history)
    write_csv(out / "loss_history.csv", history)

    first = history[0]["loss"]
    final = history[-1]["loss"]
    lines = [
        "# Feed-Forward Predictor Training",
        "",
        f"Rows: {len(rows)}",
        f"Gaussians: {int(num_gaussians)}",
        f"Epochs: {int(epochs)}",
        f"Initial loss: {first:.6f}",
        f"Final loss: {final:.6f}",
        f"Checkpoint: `{checkpoint_path}`",
        "",
        "Verdict: learned checkpoint path was trained and saved. This is a minimal "
        "contract smoke, not a speedup or generalization claim.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "checkpoint_path": str(checkpoint_path),
        "summary_path": str(out / "summary.md"),
        "loss_history_path": str(out / "loss_history.json"),
        "initial_loss": first,
        "final_loss": final,
        "rows": len(rows),
        "num_gaussians": int(num_gaussians),
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train tiny FF-001 predictor from teacher manifest")
    p.add_argument("manifest")
    p.add_argument("--outdir", default="results/feedforward_train")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--num-gaussians", type=int, default=None)
    p.add_argument("--max-scale-norm", type=float, default=0.5)
    args = p.parse_args()
    result = train_feedforward_predictor(
        args.manifest,
        outdir=args.outdir,
        image_size=args.image_size,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        num_gaussians=args.num_gaussians,
        max_scale_norm=args.max_scale_norm,
    )
    print(result)


if __name__ == "__main__":
    main()
