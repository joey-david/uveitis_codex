#!/usr/bin/env python3
"""Calibrate per-class postprocess params on val set using detection AP/F1."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
import torch

from uveitis_pipeline.common import load_yaml, read_jsonl, save_json
from uveitis_pipeline.retfound_mask import RetFoundEncoder, RetFoundMaskModel, load_retfound_vit, load_retfound_weights


def _bbox_from_poly(poly: list[float], w: int, h: int) -> list[float] | None:
    """Convert normalized polygon points to absolute xyxy box."""
    if len(poly) < 6:
        return None
    xs = [float(poly[i]) * w for i in range(0, len(poly), 2)]
    ys = [float(poly[i]) * h for i in range(1, len(poly), 2)]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _iou(a: list[float], b: list[float]) -> float:
    """Compute IoU for xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def _ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """Compute AP using 101-point interpolation."""
    out = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        p = prec[rec >= t]
        out += float(p.max()) if p.size else 0.0
    return out / 101.0


def _nms(preds: list[dict], iou_thr: float) -> list[dict]:
    """Apply class-local NMS on xyxy boxes."""
    kept: list[dict] = []
    cand = sorted(preds, key=lambda x: float(x["score"]), reverse=True)
    while cand:
        cur = cand.pop(0)
        kept.append(cur)
        nxt = []
        for p in cand:
            if _iou(cur["bbox_xyxy"], p["bbox_xyxy"]) <= iou_thr:
                nxt.append(p)
        cand = nxt
    return kept


def _extract_boxes(prob: np.ndarray, threshold: float, min_area_px: int, open_k: int, close_k: int, simplify_eps: float) -> list[dict]:
    """Extract scored component boxes from a class probability map."""
    mask = (prob >= float(threshold)).astype(np.uint8)
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    h, w = prob.shape
    out: list[dict] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(min_area_px):
            continue
        poly = cnt.reshape(-1, 2).astype(np.float32)
        if simplify_eps > 0:
            poly = cv2.approxPolyDP(poly, epsilon=float(simplify_eps), closed=True).reshape(-1, 2).astype(np.float32)
        if poly.shape[0] < 3:
            continue
        comp = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(comp, [np.round(poly).astype(np.int32)], 1)
        score = float(prob[comp > 0].mean()) if np.any(comp > 0) else float(prob[int(poly[0, 1]), int(poly[0, 0])])
        x1 = float(np.clip(poly[:, 0].min(), 0, w - 1))
        y1 = float(np.clip(poly[:, 1].min(), 0, h - 1))
        x2 = float(np.clip(poly[:, 0].max(), 0, w - 1))
        y2 = float(np.clip(poly[:, 1].max(), 0, h - 1))
        if x2 > x1 and y2 > y1:
            out.append({"score": score, "bbox_xyxy": [x1, y1, x2, y2]})
    return out


def _evaluate_class(gt_rows: list[dict], pred_rows: list[dict], iou_thr: float) -> dict:
    """Evaluate one class with AP/F1 metrics."""
    gt_items = [{"record_id": g["record_id"], "bbox": g["bbox"], "used": False} for g in gt_rows]
    gt_lookup: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(gt_items):
        gt_lookup[g["record_id"]].append(i)

    preds = sorted(pred_rows, key=lambda x: float(x["score"]), reverse=True)
    tp = np.zeros(len(preds), dtype=np.float64)
    fp = np.zeros(len(preds), dtype=np.float64)

    for i, p in enumerate(preds):
        cand = gt_lookup.get(str(p["record_id"]), [])
        best_iou = -1.0
        best_j = -1
        for j in cand:
            if gt_items[j]["used"]:
                continue
            v = _iou(p["bbox_xyxy"], gt_items[j]["bbox"])
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            tp[i] = 1.0
            gt_items[best_j]["used"] = True
        else:
            fp[i] = 1.0

    n_gt = len(gt_items)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    rec = cum_tp / max(float(n_gt), 1.0)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    ap50 = _ap(rec, prec) if len(preds) > 0 and n_gt > 0 else 0.0
    tp_n = int(tp.sum())
    fp_n = int(fp.sum())
    fn_n = max(0, n_gt - tp_n)
    precision = tp_n / max(tp_n + fp_n, 1)
    recall = tp_n / max(tp_n + fn_n, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "ap50": float(ap50),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "n_gt": int(n_gt),
        "n_pred": int(len(preds)),
        "tp": int(tp_n),
        "fp": int(fp_n),
        "fn": int(fn_n),
    }


def main() -> None:
    """Tune per-class detection threshold/min-area for stage-6 inference."""
    parser = argparse.ArgumentParser(description="Stage-7b detection-level postprocess calibration")
    parser.add_argument("--config", default="configs/stage7_calibrate_thresholds.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ckpt = torch.load(cfg["model"]["checkpoint"], map_location="cpu")
    class_names = ckpt.get("class_names") or load_yaml(cfg["model"]["class_map_active"]).get("categories", [])
    num_classes = len(class_names)

    image_size = int(cfg["model"].get("image_size", 1024))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
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

    iou_thr = float(cfg.get("search", {}).get("iou_threshold", 0.5))
    thresholds = [float(v) for v in cfg.get("search", {}).get("thresholds", [0.15, 0.2, 0.25, 0.3, 0.4, 0.5])]
    min_areas = [int(v) for v in cfg.get("search", {}).get("min_areas", [8, 16, 24, 32, 48, 64])]
    open_k = int(cfg.get("search", {}).get("open_kernel", 0))
    close_k = int(cfg.get("search", {}).get("close_kernel", 3))
    simplify_eps = float(cfg.get("search", {}).get("polygon_simplify_eps", 1.0))
    nms_iou = float(cfg.get("search", {}).get("nms_iou", 0.2))
    absent_thr = float(cfg.get("search", {}).get("absent_class_threshold", 0.95))
    default_min_area = int(cfg.get("search", {}).get("default_min_area_px", 64))
    target_metric = str(cfg.get("search", {}).get("target_metric", "ap50"))

    cache = []
    gt_by_class: dict[str, list[dict]] = {name: [] for name in class_names}

    for row in rows:
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        inp = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        ten = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            prob = torch.sigmoid(model(ten))[0].detach().cpu().numpy()

        probs_resized = [cv2.resize(prob[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(num_classes)]
        cache.append({"record_id": str(row.get("record_id", "")), "probs": probs_resized, "width": w, "height": h})

        gt = json.loads(Path(row["labels_path"]).read_text(encoding="utf-8"))
        for obj in gt.get("objects", []):
            name = str(obj.get("class_name", ""))
            if name not in gt_by_class:
                continue
            box = _bbox_from_poly(obj["polygon"], w=int(gt.get("width", w)), h=int(gt.get("height", h)))
            if box is not None:
                gt_by_class[name].append({"record_id": str(row.get("record_id", "")), "bbox": box})

    out = {
        "search": {
            "iou_threshold": iou_thr,
            "thresholds": thresholds,
            "min_areas": min_areas,
            "open_kernel": open_k,
            "close_kernel": close_k,
            "polygon_simplify_eps": simplify_eps,
            "nms_iou": nms_iou,
            "target_metric": target_metric,
        },
        "classes": {},
    }

    for c, name in enumerate(class_names):
        gt_rows = gt_by_class.get(name, [])
        if not gt_rows:
            out["classes"][name] = {
                "threshold": absent_thr,
                "min_area_px": default_min_area,
                "metric": 0.0,
                "ap50": 0.0,
                "f1": 0.0,
                "n_gt": 0,
                "n_pred": 0,
            }
            continue

        best = None
        for thr in thresholds:
            for area in min_areas:
                pred_rows = []
                for row in cache:
                    preds = _extract_boxes(
                        prob=row["probs"][c],
                        threshold=thr,
                        min_area_px=area,
                        open_k=open_k,
                        close_k=close_k,
                        simplify_eps=simplify_eps,
                    )
                    preds = _nms(preds, iou_thr=nms_iou)
                    for p in preds:
                        pred_rows.append({"record_id": row["record_id"], **p})

                m = _evaluate_class(gt_rows=gt_rows, pred_rows=pred_rows, iou_thr=iou_thr)
                score = float(m["f1"] if target_metric == "f1" else m["ap50"])
                row = {
                    "threshold": float(thr),
                    "min_area_px": int(area),
                    "metric": score,
                    **m,
                }
                if best is None or row["metric"] > best["metric"] or (
                    row["metric"] == best["metric"] and row["ap50"] > best["ap50"]
                ):
                    best = row

        out["classes"][name] = best

    out_path = Path(cfg["output"].get("detection_postprocess_json", "runs/detection_postprocess.json"))
    save_json(out_path, out)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
