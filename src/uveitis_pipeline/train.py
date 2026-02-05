from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou

from .common import ensure_dir, save_json, set_seed
from .dataset import CocoDetectionDataset, collate_fn
from .modeling import build_detector


def _match_detections(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_thresh):
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    classes = sorted(set(gt_labels.tolist()) | set(pred_labels.tolist()))
    for cls in classes:
        p_idx = torch.where(pred_labels == cls)[0]
        g_idx = torch.where(gt_labels == cls)[0]
        if len(g_idx) == 0:
            per_class[cls]["fp"] += int(len(p_idx))
            continue
        if len(p_idx) == 0:
            per_class[cls]["fn"] += int(len(g_idx))
            continue

        boxes_p = pred_boxes[p_idx]
        scores_p = pred_scores[p_idx]
        order = torch.argsort(scores_p, descending=True)
        boxes_p = boxes_p[order]

        boxes_g = gt_boxes[g_idx]
        used = torch.zeros(len(boxes_g), dtype=torch.bool)
        for b in boxes_p:
            ious = box_iou(b.unsqueeze(0), boxes_g).squeeze(0)
            best = int(torch.argmax(ious))
            if ious[best] >= iou_thresh and not used[best]:
                per_class[cls]["tp"] += 1
                used[best] = True
            else:
                per_class[cls]["fp"] += 1
        per_class[cls]["fn"] += int((~used).sum())
    return per_class


def _evaluate(model, loader, device, iou_thresh, fp_targets):
    model.eval()
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    fp_scores = defaultdict(list)
    n_images = 0

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                n_images += 1
                gt_boxes = tgt["boxes"].cpu()
                gt_labels = tgt["labels"].cpu()
                pred_boxes = out["boxes"].cpu()
                pred_scores = out["scores"].cpu()
                pred_labels = out["labels"].cpu()

                stats = _match_detections(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_thresh)
                for cls, s in stats.items():
                    for k, v in s.items():
                        class_stats[cls][k] += v

                # Gather scores for FP/image operating points.
                for cls in torch.unique(pred_labels).tolist():
                    cls = int(cls)
                    p_idx = torch.where(pred_labels == cls)[0]
                    for i in p_idx:
                        fp_scores[cls].append(float(pred_scores[i]))

    metrics = {"per_class": {}, "fp_per_image": {}, "sensitivity_at_fp_per_image": {}}
    ap_like = []
    for cls, s in class_stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        ap_like.append(prec * rec)
        metrics["per_class"][str(cls)] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
        }
        metrics["fp_per_image"][str(cls)] = fp / max(n_images, 1)

    metrics["mAP_proxy"] = float(np.mean(ap_like)) if ap_like else 0.0

    for target in fp_targets:
        target = float(target)
        per_class_rec = {}
        for cls, s in class_stats.items():
            fp_pi = s["fp"] / max(n_images, 1)
            rec = s["tp"] / max(s["tp"] + s["fn"], 1)
            per_class_rec[str(cls)] = rec if fp_pi <= target else 0.0
        metrics["sensitivity_at_fp_per_image"][str(target)] = per_class_rec
    return metrics


def train_from_config(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 42)))

    run_dir = Path(cfg["run"]["output_dir"]) / cfg["run"]["name"]
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    writer = None
    if cfg["run"].get("tensorboard", True):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter((run_dir / "tb").as_posix())
    wandb_run = None
    if cfg["run"].get("wandb", False):
        import wandb

        wandb_run = wandb.init(
            project=cfg["run"].get("wandb_project", "uveitis_codex"),
            name=cfg["run"]["name"],
            config=cfg,
        )

    train_ds = CocoDetectionDataset(cfg["data"]["train_coco"], resize=cfg["data"].get("resize"))
    val_ds = CocoDetectionDataset(cfg["data"]["val_coco"], resize=cfg["data"].get("resize"))

    overfit_n = int(cfg["training"].get("overfit_num_images", 0))
    if overfit_n > 0:
        idx = list(range(min(overfit_n, len(train_ds))))
        train_ds = Subset(train_ds, idx)
        val_ds = Subset(val_ds, idx)
    else:
        max_train = int(cfg["training"].get("max_train_images", 0))
        max_val = int(cfg["training"].get("max_val_images", 0))
        seed = int(cfg.get("seed", 42))
        if max_train > 0 and max_train < len(train_ds):
            rs = np.random.RandomState(seed)
            train_ds = Subset(train_ds, rs.choice(len(train_ds), size=max_train, replace=False).tolist())
        if max_val > 0 and max_val < len(val_ds):
            rs = np.random.RandomState(seed + 1)
            val_ds = Subset(val_ds, rs.choice(len(val_ds), size=max_val, replace=False).tolist())

    sampler = None
    shuffle = True
    if cfg["training"].get("class_balanced_sampling", False) and hasattr(train_ds, "build_class_balanced_sampler"):
        sampler = train_ds.build_class_balanced_sampler()
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"].get("eval_batch_size", 2)),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        collate_fn=collate_fn,
    )

    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_detector(cfg).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    amp_dtype = str(cfg["training"].get("amp_dtype", "bf16")).lower()
    autocast_dtype = torch.bfloat16 if amp_dtype in ("bf16", "bfloat16") else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and autocast_dtype == torch.float16)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg["training"]["epochs"])),
        eta_min=float(cfg["training"].get("min_lr", 1e-6)),
    )

    if cfg["training"].get("init_checkpoint"):
        state = torch.load(cfg["training"]["init_checkpoint"], map_location="cpu")
        model.load_state_dict(state["model"], strict=False)

    best_score = -1.0
    metrics_path = run_dir / "metrics.jsonl"
    fp_targets = cfg["training"].get("fp_image_targets", [0.5, 1.0, 2.0])

    freeze_epochs = int(cfg["training"].get("freeze_epochs", 0))

    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        if epoch == freeze_epochs and hasattr(model.backbone, "set_freeze_blocks"):
            model.backbone.set_freeze_blocks(0)

        t0 = time.time()
        losses_epoch = []
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            losses_epoch.append(float(loss.item()))

        scheduler.step()
        train_loss = float(np.mean(losses_epoch)) if losses_epoch else 0.0
        val_metrics = _evaluate(
            model,
            val_loader,
            device,
            iou_thresh=float(cfg["training"].get("eval_iou", 0.3)),
            fp_targets=fp_targets,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mAP_proxy": val_metrics["mAP_proxy"],
            "elapsed_sec": time.time() - t0,
            "lr": optimizer.param_groups[0]["lr"],
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/mAP_proxy", val_metrics["mAP_proxy"], epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        if wandb_run is not None:
            wandb_run.log(row)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "config": cfg,
            "metrics": row,
            "val_metrics": val_metrics,
        }
        if bool(cfg["run"].get("save_optimizer", False)):
            ckpt["optimizer"] = optimizer.state_dict()

        save_epochs = int(cfg["run"].get("save_every_epochs", 0))
        save_all = bool(cfg["run"].get("save_epoch_checkpoints", False))
        if save_all or (save_epochs > 0 and (epoch % save_epochs == 0)):
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pth")
        torch.save(ckpt, ckpt_dir / "last.pth")

        if val_metrics["mAP_proxy"] > best_score:
            best_score = val_metrics["mAP_proxy"]
            torch.save(ckpt, ckpt_dir / "best.pth")

    if writer is not None:
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    best = torch.load(ckpt_dir / "best.pth", map_location="cpu")
    report = {
        "best_epoch": int(best["epoch"]),
        "best_val_mAP_proxy": float(best["metrics"]["val_mAP_proxy"]),
        "val_metrics": best["val_metrics"],
    }
    save_json(run_dir / "val_report.json", report)
    return report
