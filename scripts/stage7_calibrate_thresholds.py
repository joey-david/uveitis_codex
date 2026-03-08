#!/usr/bin/env python3
"""Calibrate per-class mask thresholds on validation native labels."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

import cv2
import numpy as np
import torch

from uveitis_pipeline.common import load_yaml, read_jsonl, save_json
from uveitis_pipeline.retfound_mask import RetFoundEncoder, RetFoundMaskModel, load_retfound_vit, load_retfound_weights


def _rasterize_gt(labels_path: str, num_classes: int, h: int, w: int) -> np.ndarray:
    """Rasterize native polygon labels to CxHxW ground-truth masks."""
    rec = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    mask = np.zeros((num_classes, h, w), dtype=np.uint8)
    for obj in rec.get("objects", []):
        cid = int(obj["class_id"]) - 1
        if cid < 0 or cid >= num_classes:
            continue
        poly = np.array(list(zip(obj["polygon"][0::2], obj["polygon"][1::2])), dtype=np.float32)
        poly[:, 0] = np.clip(poly[:, 0] * w, 0, max(0, w - 1))
        poly[:, 1] = np.clip(poly[:, 1] * h, 0, max(0, h - 1))
        cv2.fillPoly(mask[cid], [np.round(poly).astype(np.int32)], 1)
    return mask


def main() -> None:
    """Search per-class thresholds that maximize pixel IoU on validation data."""
    parser = argparse.ArgumentParser(description="Stage-7 per-class threshold calibration")
    parser.add_argument("--config", default="configs/stage7_calibrate_thresholds.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ckpt = torch.load(cfg["model"]["checkpoint"], map_location="cpu")
    class_names = ckpt.get("class_names") or load_yaml(cfg["model"]["class_map_active"]).get("categories", [])
    num_classes = len(class_names)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    image_size = int(cfg["model"].get("image_size", 1024))

    vit = load_retfound_vit(cfg["model"]["vendor_dir"], image_size=image_size)
    load_retfound_weights(vit, cfg["model"]["retfound_ckpt"])
    model = RetFoundMaskModel(
        encoder=RetFoundEncoder(vit),
        num_classes=num_classes,
        decoder_channels=int(cfg["model"].get("decoder_channels", 256)),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    rows = read_jsonl(cfg["data"]["val_index"])
    max_images = int(cfg["data"].get("max_images", 0))
    if max_images > 0:
        rows = rows[:max_images]

    thresholds = [float(t) for t in cfg.get("search", {}).get("thresholds", [0.2, 0.3, 0.4, 0.5, 0.6])]
    t_count = len(thresholds)
    tp = np.zeros((num_classes, t_count), dtype=np.float64)
    fp = np.zeros((num_classes, t_count), dtype=np.float64)
    fn = np.zeros((num_classes, t_count), dtype=np.float64)
    gt_pixels = np.zeros((num_classes,), dtype=np.float64)

    for row in rows:
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        inp = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        ten = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        ten = ten.to(device)
        with torch.no_grad():
            logits = model(ten)
            prob = torch.sigmoid(logits)[0].cpu().numpy()

        gt = _rasterize_gt(row["labels_path"], num_classes=num_classes, h=h, w=w)
        gt_pixels += gt.reshape(num_classes, -1).sum(axis=1)
        for c in range(num_classes):
            pr = cv2.resize(prob[c], (w, h), interpolation=cv2.INTER_LINEAR)
            gt_c = gt[c] > 0
            for j, thr in enumerate(thresholds):
                pd = pr >= thr
                tp[c, j] += float(np.logical_and(pd, gt_c).sum())
                fp[c, j] += float(np.logical_and(pd, ~gt_c).sum())
                fn[c, j] += float(np.logical_and(~pd, gt_c).sum())

    out = {"thresholds": thresholds, "classes": {}}
    absent_thr = float(cfg.get("search", {}).get("absent_class_threshold", 0.95))
    for c, name in enumerate(class_names):
        iou = tp[c] / np.maximum(tp[c] + fp[c] + fn[c], 1e-6)
        if gt_pixels[c] <= 0:
            out["classes"][name] = {
                "best_threshold": absent_thr,
                "best_iou": 0.0,
                "iou_by_threshold": {str(t): float(v) for t, v in zip(thresholds, iou.tolist())},
                "gt_pixels": 0,
            }
            continue
        best_idx = int(np.argmax(iou))
        out["classes"][name] = {
            "best_threshold": thresholds[best_idx],
            "best_iou": float(iou[best_idx]),
            "iou_by_threshold": {str(t): float(v) for t, v in zip(thresholds, iou.tolist())},
            "gt_pixels": int(gt_pixels[c]),
        }

    out_path = Path(cfg["output"].get("thresholds_json", "runs/mask_thresholds.json"))
    save_json(out_path, out)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
