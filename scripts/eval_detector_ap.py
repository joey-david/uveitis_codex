#!/usr/bin/env python3
from __future__ import annotations

"""
Compute a real (score-swept) AP/mAP for our detector checkpoints.

This is intentionally lightweight: no pycocotools/torchmetrics dependency.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uveitis_pipeline.dataset import CocoDetectionDataset, collate_fn  # noqa: E402
from uveitis_pipeline.modeling import build_detector  # noqa: E402


def _ap_from_scores(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    if n_gt <= 0:
        return float("nan")
    tp = tp.astype(np.float64)
    fp = fp.astype(np.float64)
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    rec = tp_c / max(n_gt, 1)
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)

    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def _eval_ap(
    model,
    loader,
    device: torch.device,
    iou_thresh: float,
    max_dets: int,
    class_ids: list[int] | None = None,
) -> dict:
    model.eval()
    n_gt = defaultdict(int)
    pred_scores = defaultdict(list)
    pred_tp = defaultdict(list)

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                gt_boxes = tgt["boxes"].cpu()
                gt_labels = tgt["labels"].cpu()
                for cls in torch.unique(gt_labels).tolist():
                    n_gt[int(cls)] += int((gt_labels == cls).sum())

                scores = out["scores"].detach().cpu()
                keep = torch.argsort(scores, descending=True)[: int(max_dets)]
                pb = out["boxes"].detach().cpu()[keep]
                ps = scores[keep]
                pl = out["labels"].detach().cpu()[keep]

                classes = sorted(set(gt_labels.tolist()) | set(pl.tolist()))
                for cls in classes:
                    cls = int(cls)
                    p_idx = torch.where(pl == cls)[0]
                    g_idx = torch.where(gt_labels == cls)[0]
                    if len(p_idx) == 0:
                        continue
                    boxes_p = pb[p_idx]
                    scores_p = ps[p_idx]
                    order = torch.argsort(scores_p, descending=True)
                    boxes_p = boxes_p[order]
                    scores_p = scores_p[order]

                    if len(g_idx) == 0:
                        for s in scores_p.tolist():
                            pred_scores[cls].append(float(s))
                            pred_tp[cls].append(0)
                        continue

                    boxes_g = gt_boxes[g_idx]
                    used = torch.zeros(len(boxes_g), dtype=torch.bool)
                    ious = box_iou(boxes_p, boxes_g)
                    for i in range(len(boxes_p)):
                        best = int(torch.argmax(ious[i]))
                        match = float(ious[i, best]) >= float(iou_thresh) and not bool(used[best])
                        used[best] = used[best] | match
                        pred_scores[cls].append(float(scores_p[i]))
                        pred_tp[cls].append(1 if match else 0)

    if class_ids is None:
        class_ids = sorted(set(n_gt.keys()) | set(pred_scores.keys()))

    ap_per_class: dict[str, float] = {}
    for cls in class_ids:
        cls = int(cls)
        ng = int(n_gt.get(cls, 0))
        scores = pred_scores.get(cls, [])
        tps = pred_tp.get(cls, [])
        if ng > 0 and not scores:
            ap_per_class[str(cls)] = 0.0
            continue
        if ng == 0:
            ap_per_class[str(cls)] = float("nan")
            continue
        scores = np.asarray(scores, dtype=np.float64)
        tp = np.asarray(tps, dtype=np.int64)
        order = np.argsort(-scores)
        tp = tp[order]
        fp = 1 - tp
        ap_per_class[str(cls)] = _ap_from_scores(tp, fp, ng)

    aps = [v for v in ap_per_class.values() if not np.isnan(v)]
    return {"mAP": float(np.mean(aps)) if aps else 0.0, "ap_per_class": ap_per_class, "gt_per_class": dict(n_gt)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute score-swept AP/mAP for a checkpoint on a COCO val set.")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--resize", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-dets", type=int, default=400)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    cfg = state.get("config", {})
    cfg = dict(cfg)
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg["data"]["val_coco"] = args.coco.as_posix()
    if args.resize > 0:
        cfg["data"]["resize"] = int(args.resize)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_detector(cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    # For AP computation we want a wide score range; rely on score-sorting, not internal filtering.
    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "score_thresh"):
        model.roi_heads.score_thresh = 0.0
    if hasattr(model, "score_thresh"):
        model.score_thresh = 0.0
    if hasattr(model, "detections_per_img"):
        model.detections_per_img = int(args.max_dets)
    model.eval()

    ds = CocoDetectionDataset(args.coco, resize=cfg["data"].get("resize"))
    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    class_ids = [int(c["id"]) for c in coco.get("categories", [])]
    if args.num_images > 0 and args.num_images < len(ds):
        ds = Subset(ds, list(range(int(args.num_images))))
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    out = _eval_ap(
        model,
        loader,
        device,
        iou_thresh=float(args.iou),
        max_dets=int(args.max_dets),
        class_ids=class_ids or None,
    )
    print({"iou": float(args.iou), **out})


if __name__ == "__main__":
    main()
