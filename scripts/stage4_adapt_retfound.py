#!/usr/bin/env python3
"""Contrastive adaptation of RETFound features on masked/normalized fundus images."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from uveitis_pipeline.common import load_yaml, read_jsonl, save_json, set_seed
from uveitis_pipeline.native_dataset import AdaptImageDataset
from uveitis_pipeline.retfound_mask import (
    ContrastiveProjectionHead,
    RetFoundEncoder,
    load_retfound_vit,
    load_retfound_weights,
    nt_xent_loss,
)


def _collect_paths(cfg: dict) -> list[str]:
    """Collect preprocessed image paths for adaptation."""
    rows: list[dict] = []
    for m in cfg["data"]["manifests"]:
        rows.extend(read_jsonl(m))

    split_ids = None
    split_json = cfg["data"].get("split_json")
    split_names = cfg["data"].get("splits", ["train"])
    if split_json:
        split = json.loads(Path(split_json).read_text(encoding="utf-8"))
        keep: set[str] = set()
        for s in split_names:
            keep.update(split.get(s, []))
        split_ids = keep

    keep_datasets = set(cfg["data"].get("datasets", []))
    preproc_root = Path(cfg["data"]["preproc_root"])
    image_dir = preproc_root / cfg["data"].get("image_subdir", "norm")

    out: list[str] = []
    for row in rows:
        if keep_datasets and row.get("dataset") not in keep_datasets:
            continue
        if split_ids is not None and row["image_id"] not in split_ids:
            continue
        key = row["image_id"].replace("::", "__")
        p = image_dir / f"{key}.png"
        if p.exists():
            out.append(p.as_posix())

    max_images = int(cfg["data"].get("max_images", 0))
    if max_images > 0:
        out = out[:max_images]
    return out


def main() -> None:
    """Train and save adapted RETFound encoder checkpoint."""
    parser = argparse.ArgumentParser(description="Stage-4 RETFound contrastive adaptation")
    parser.add_argument("--config", default="configs/stage4_adapt_retfound.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    run_dir = Path(cfg["run"]["output_dir"]) / cfg["run"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    paths = _collect_paths(cfg)
    if not paths:
        raise RuntimeError("No adaptation images found. Check manifests/splits/preproc root.")

    ds = AdaptImageDataset(paths, image_size=int(cfg["model"].get("input_size", 224)))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        drop_last=True,
    )

    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    vit = load_retfound_vit(cfg["model"]["vendor_dir"], image_size=int(cfg["model"].get("input_size", 224)))
    load_retfound_weights(vit, cfg["model"]["retfound_ckpt"])

    encoder = RetFoundEncoder(vit).to(device)
    encoder.set_freeze_blocks(int(cfg["train"].get("freeze_blocks", 8)))
    proj = ContrastiveProjectionHead(encoder.embed_dim, proj_dim=int(cfg["model"].get("proj_dim", 256))).to(device)

    params = [p for p in list(encoder.parameters()) + list(proj.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = 1e9
    metrics: list[dict] = []
    epochs = int(cfg["train"].get("epochs", 10))
    temp = float(cfg["train"].get("temperature", 0.2))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        encoder.train()
        proj.train()

        for v1, v2 in loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                f1 = encoder(v1).mean(dim=(2, 3))
                f2 = encoder(v2).mean(dim=(2, 3))
                z1 = proj(f1)
                z2 = proj(f2)
                loss = nt_xent_loss(z1, z2, temperature=temp)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            losses.append(float(loss.item()))

        epoch_loss = float(np.mean(losses)) if losses else 0.0
        row = {"epoch": epoch, "loss": epoch_loss, "time_s": time.time() - t0}
        metrics.append(row)
        print(row)

        ckpt = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "projection": proj.state_dict(),
            "cfg": cfg,
        }
        torch.save(ckpt, run_dir / "last.pt")
        if epoch_loss < best:
            best = epoch_loss
            torch.save(ckpt, run_dir / "best.pt")

    save_json(run_dir / "metrics.json", {"best_loss": best, "epochs": metrics})
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
