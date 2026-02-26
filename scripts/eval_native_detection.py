#!/usr/bin/env python3
"""Evaluate native detection predictions against native GT using bbox AP/F1."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uveitis_pipeline.common import read_jsonl, save_json


def _bbox_from_polygon(poly_norm: list[float], w: int, h: int) -> list[float] | None:
    """Convert normalized polygon list to absolute xyxy box."""
    if len(poly_norm) < 6:
        return None
    xs = [float(poly_norm[i]) * w for i in range(0, len(poly_norm), 2)]
    ys = [float(poly_norm[i]) * h for i in range(1, len(poly_norm), 2)]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two xyxy boxes."""
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


def _ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    """Compute AP via 101-point interpolation."""
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        p = prec[rec >= t]
        ap += float(p.max()) if p.size else 0.0
    return ap / 101.0


def evaluate(gt_index: str, pred_jsonl: str, iou_thr: float, score_thr: float) -> dict:
    """Evaluate predictions against GT and return aggregate + per-class metrics."""
    gt_rows = read_jsonl(gt_index)
    pred_rows = read_jsonl(pred_jsonl)
    pred_by_record = {str(r.get("record_id", "")): r for r in pred_rows}

    gt_by_class: dict[str, list[dict]] = defaultdict(list)
    pred_by_class: dict[str, list[dict]] = defaultdict(list)

    class_names = set()
    for row in gt_rows:
        rec_id = str(row.get("record_id", ""))
        gt = json.loads(Path(row["labels_path"]).read_text(encoding="utf-8"))
        w = int(gt.get("width", row.get("width", 0)))
        h = int(gt.get("height", row.get("height", 0)))
        for obj in gt.get("objects", []):
            box = _bbox_from_polygon(obj["polygon"], w, h)
            if box is None:
                continue
            name = str(obj["class_name"])
            class_names.add(name)
            gt_by_class[name].append({"record_id": rec_id, "bbox": box, "used": False})

        pred_rec = pred_by_record.get(rec_id, {"predictions": []})
        for p in pred_rec.get("predictions", []):
            name = str(p.get("class_name", ""))
            if not name:
                continue
            class_names.add(name)
            score = float(p.get("score", 0.0))
            if score < score_thr:
                continue
            box = p.get("bbox_xyxy")
            if not box or len(box) != 4:
                continue
            pred_by_class[name].append({"record_id": rec_id, "bbox": [float(v) for v in box], "score": score})

    per_class = {}
    ap_vals = []
    ap_vals_all = []
    f1_vals = []
    f1_vals_all = []

    for name in sorted(class_names):
        gt_items = gt_by_class.get(name, [])
        preds = sorted(pred_by_class.get(name, []), key=lambda x: x["score"], reverse=True)
        n_gt = len(gt_items)
        if n_gt == 0 and len(preds) == 0:
            continue

        gt_lookup = defaultdict(list)
        for i, g in enumerate(gt_items):
            gt_lookup[g["record_id"]].append(i)

        tp = np.zeros(len(preds), dtype=np.float64)
        fp = np.zeros(len(preds), dtype=np.float64)

        for i, p in enumerate(preds):
            cand = gt_lookup.get(p["record_id"], [])
            best_iou = -1.0
            best_j = -1
            for j in cand:
                if gt_items[j]["used"]:
                    continue
                v = _iou(p["bbox"], gt_items[j]["bbox"])
                if v > best_iou:
                    best_iou = v
                    best_j = j
            if best_j >= 0 and best_iou >= iou_thr:
                tp[i] = 1.0
                gt_items[best_j]["used"] = True
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec = cum_tp / max(float(n_gt), 1.0)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

        ap = _ap_from_pr(rec, prec) if len(preds) > 0 and n_gt > 0 else 0.0
        ap_vals_all.append(ap)
        if n_gt > 0:
            ap_vals.append(ap)

        tp_n = int(tp.sum())
        fp_n = int(fp.sum())
        fn_n = max(0, n_gt - tp_n)
        precision = tp_n / max(tp_n + fp_n, 1)
        recall = tp_n / max(tp_n + fn_n, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        f1_vals_all.append(f1)
        if n_gt > 0:
            f1_vals.append(f1)

        per_class[name] = {
            "n_gt": n_gt,
            "n_pred": len(preds),
            "tp": tp_n,
            "fp": fp_n,
            "fn": fn_n,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": ap,
        }

    return {
        "iou_threshold": iou_thr,
        "score_threshold": score_thr,
        "map50": float(np.mean(ap_vals)) if ap_vals else 0.0,
        "map50_all_classes": float(np.mean(ap_vals_all)) if ap_vals_all else 0.0,
        "macro_f1": float(np.mean(f1_vals)) if f1_vals else 0.0,
        "macro_f1_all_classes": float(np.mean(f1_vals_all)) if f1_vals_all else 0.0,
        "per_class": per_class,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate native predictions")
    parser.add_argument("--gt-index", required=True)
    parser.add_argument("--pred-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--score", type=float, default=0.0)
    args = parser.parse_args()

    metrics = evaluate(args.gt_index, args.pred_jsonl, iou_thr=args.iou, score_thr=args.score)
    save_json(args.out, metrics)
    print(json.dumps({"out": args.out, "map50": metrics["map50"], "macro_f1": metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
