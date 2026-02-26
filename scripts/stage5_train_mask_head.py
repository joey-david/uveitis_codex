#!/usr/bin/env python3
"""Train RETFound mask-first lesion model on native polygon labels."""

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

from uveitis_pipeline.common import load_yaml, save_json, set_seed
from uveitis_pipeline.native_dataset import NativeMaskDataset, mask_collate
from uveitis_pipeline.retfound_mask import (
    RetFoundEncoder,
    RetFoundMaskModel,
    load_encoder_state,
    load_retfound_vit,
    load_retfound_weights,
    masked_bce_dice_loss,
    multi_label_iou,
)


def _load_num_classes(class_map_path: str) -> tuple[int, list[str]]:
    """Read active class map and return class count + names."""
    data = load_yaml(class_map_path)
    classes = [str(c) for c in data.get("categories", [])]
    if not classes:
        raise ValueError(f"No classes found in {class_map_path}")
    return len(classes), classes


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    class_weights: torch.Tensor | None = None,
) -> dict:
    """Evaluate mean loss and IoU on validation set."""
    model.eval()
    losses = []
    ious = []
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            losses.append(float(masked_bce_dice_loss(logits, masks, class_weights=class_weights).item()))
            ious.append(multi_label_iou(logits, masks, threshold=threshold).detach().cpu().numpy())

    iou_arr = np.stack(ious, axis=0) if ious else np.zeros((1, 1), dtype=np.float32)
    per_class = iou_arr.mean(axis=0).tolist()
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "mean_iou": float(np.mean(per_class)) if per_class else 0.0,
        "per_class_iou": [float(v) for v in per_class],
    }


def _compute_class_weights(
    index_jsonl: str,
    num_classes: int,
    power: float,
    min_w: float,
    max_w: float,
) -> list[float]:
    """Estimate inverse-frequency class weights from training annotations."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for line in Path(index_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rec = json.loads(Path(row["labels_path"]).read_text(encoding="utf-8"))
        seen = {int(obj["class_id"]) - 1 for obj in rec.get("objects", []) if int(obj["class_id"]) >= 1}
        for c in seen:
            if 0 <= c < num_classes:
                counts[c] += 1.0
    inv = ((counts.max() + 1.0) / (counts + 1.0)) ** float(power)
    inv = np.clip(inv, float(min_w), float(max_w))
    inv = inv / max(inv.mean(), 1e-6)
    return [float(v) for v in inv.tolist()]


def main() -> None:
    """Train mask model and save best/last checkpoints with metrics."""
    parser = argparse.ArgumentParser(description="Stage-5 train RETFound mask model")
    parser.add_argument("--config", default="configs/stage5_train_mask_head.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    run_dir = Path(cfg["run"]["output_dir"]) / cfg["run"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    num_classes, class_names = _load_num_classes(cfg["data"]["class_map_active"])

    train_ds = NativeMaskDataset(
        index_jsonl=cfg["data"]["train_index"],
        num_classes=num_classes,
        image_size=int(cfg["data"].get("image_size", 1024)),
        keep_empty=bool(cfg["data"].get("train_keep_empty", cfg["data"].get("keep_empty", True))),
    )
    val_ds = NativeMaskDataset(
        index_jsonl=cfg["data"]["val_index"],
        num_classes=num_classes,
        image_size=int(cfg["data"].get("image_size", 1024)),
        keep_empty=bool(cfg["data"].get("val_keep_empty", True)),
    )

    sampler = (
        train_ds.build_balanced_sampler(
            mode=str(cfg["train"].get("sampler_mode", "mean_inv")),
            power=float(cfg["train"].get("sampler_power", 1.0)),
            empty_weight=float(cfg["train"].get("empty_sample_weight", 0.25)),
        )
        if bool(cfg["train"].get("balanced_sampler", True))
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 2)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        collate_fn=mask_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"].get("eval_batch_size", 2)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        collate_fn=mask_collate,
    )

    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    class_weights_t: torch.Tensor | None = None
    if bool(cfg["train"].get("auto_class_weights", False)):
        cw = _compute_class_weights(
            index_jsonl=cfg["data"]["train_index"],
            num_classes=num_classes,
            power=float(cfg["train"].get("class_weight_power", 1.0)),
            min_w=float(cfg["train"].get("class_weight_min", 0.5)),
            max_w=float(cfg["train"].get("class_weight_max", 8.0)),
        )
        print({"class_weights": {n: float(v) for n, v in zip(class_names, cw)}})
        class_weights_t = torch.tensor(cw, dtype=torch.float32, device=device)

    vit = load_retfound_vit(cfg["model"]["vendor_dir"], image_size=int(cfg["data"].get("image_size", 1024)))
    load_retfound_weights(vit, cfg["model"]["retfound_ckpt"])
    encoder = RetFoundEncoder(vit)

    adapt_ckpt = cfg["model"].get("adapt_ckpt")
    if adapt_ckpt:
        state = torch.load(adapt_ckpt, map_location="cpu")
        if "encoder" in state:
            enc_state = state["encoder"]
        elif "model" in state and isinstance(state["model"], dict):
            model_sd = state["model"]
            if any(str(k).startswith("encoder.") for k in model_sd.keys()):
                enc_state = {str(k).replace("encoder.", "", 1): v for k, v in model_sd.items() if str(k).startswith("encoder.")}
            else:
                enc_state = model_sd
        else:
            enc_state = state
        load_encoder_state(encoder, enc_state)

    encoder.set_freeze_blocks(int(cfg["train"].get("freeze_blocks", 8)))
    model = RetFoundMaskModel(
        encoder=encoder,
        num_classes=num_classes,
        decoder_channels=int(cfg["model"].get("decoder_channels", 256)),
    ).to(device)
    init_ckpt = cfg["model"].get("init_ckpt")
    if init_ckpt:
        init_state = torch.load(init_ckpt, map_location="cpu")
        if "model" in init_state and isinstance(init_state["model"], dict):
            model.load_state_dict(init_state["model"], strict=False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, int(cfg["train"].get("epochs", 30))),
        eta_min=float(cfg["train"].get("min_lr", 1e-6)),
    )

    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = -1.0
    threshold = float(cfg["eval"].get("mask_threshold", 0.5))
    focal_gamma = float(cfg["train"].get("focal_gamma", 0.0))
    focal_alpha = float(cfg["train"].get("focal_alpha", 0.25))
    history: list[dict] = []

    for epoch in range(1, int(cfg["train"].get("epochs", 30)) + 1):
        t0 = time.time()
        model.train()
        losses = []

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images)
                loss = masked_bce_dice_loss(
                    logits,
                    masks,
                    class_weights=class_weights_t,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            losses.append(float(loss.item()))

        sched.step()
        train_loss = float(np.mean(losses)) if losses else 0.0
        val_metrics = _evaluate(
            model,
            val_loader,
            device=device,
            threshold=threshold,
            class_weights=class_weights_t,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_per_class_iou": val_metrics["per_class_iou"],
            "time_s": time.time() - t0,
        }
        history.append(row)
        print(row)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "class_names": class_names,
            "cfg": cfg,
        }
        torch.save(ckpt, run_dir / "last.pt")
        if val_metrics["mean_iou"] > best:
            best = val_metrics["mean_iou"]
            torch.save(ckpt, run_dir / "best.pt")

    save_json(run_dir / "metrics.json", {"best_val_mean_iou": best, "epochs": history, "class_names": class_names})
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
