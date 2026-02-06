#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uveitis_pipeline.vascularite import bbox_to_poly_px, obb_norm_to_poly_px


def _poly_area(poly: np.ndarray) -> float:
    return float(abs(cv2.contourArea(poly.astype(np.float32))))


def _as_hull(poly: np.ndarray) -> np.ndarray:
    poly = poly.astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(poly, returnPoints=True)
    return hull.reshape(-1, 2).astype(np.float32)


def _poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = _as_hull(a)
    b = _as_hull(b)
    ia, _ = cv2.intersectConvexConvex(a, b)
    if ia <= 0:
        return 0.0
    ua = _poly_area(a) + _poly_area(b) - float(ia)
    return float(ia) / max(ua, 1e-6)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    # VOC-style: precision envelope + integral over recall.
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True, help="Dir containing per-image json predictions with 'obb' and 'score'")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--min-score", type=float, default=0.05)
    args = ap.parse_args()

    coco = _load_json(args.coco)
    cats = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    vascular_ids = {cid for cid, name in cats.items() if name == "vascularite"}
    if not vascular_ids:
        raise ValueError("No category named 'vascularite' in COCO categories")

    images = {int(im["id"]): im for im in coco.get("images", [])}
    by_img = defaultdict(list)
    for a in coco.get("annotations", []):
        if int(a["category_id"]) in vascular_ids:
            by_img[int(a["image_id"])].append(a)

    gts = {}
    for img_id, anns in by_img.items():
        im = images[img_id]
        w = int(im["width"])
        h = int(im["height"])
        polys = []
        for a in anns:
            obb = a.get("obb")
            if isinstance(obb, list) and len(obb) == 8:
                polys.append(obb_norm_to_poly_px([float(v) for v in obb], w=w, h=h))
            else:
                polys.append(bbox_to_poly_px(a["bbox"]))
        gts[img_id] = polys

    preds_all = []
    for p in sorted(args.preds.glob("*.json")):
        obj = _load_json(p)
        image_id = obj.get("image_id")
        # If missing, try to match by filename stem (COCO 'image_id' holds original string id sometimes).
        if image_id is None:
            continue
        # Find COCO numeric image id
        coco_img_id = None
        for k, im in images.items():
            if im.get("image_id") == image_id:
                coco_img_id = k
                break
        if coco_img_id is None:
            continue
        im = images[coco_img_id]
        w = int(im["width"])
        h = int(im["height"])
        for pr in obj.get("predictions", []):
            sc = float(pr.get("score", 0.0))
            if sc < float(args.min_score):
                continue
            obb = pr.get("obb")
            if not (isinstance(obb, list) and len(obb) == 8):
                continue
            poly = obb_norm_to_poly_px([float(v) for v in obb], w=w, h=h)
            preds_all.append((coco_img_id, sc, poly))

    preds_all.sort(key=lambda t: t[1], reverse=True)
    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        raise RuntimeError("No vascularite GT instances found")

    used = {img_id: np.zeros(len(polys), dtype=bool) for img_id, polys in gts.items()}
    tp = np.zeros(len(preds_all), dtype=np.float32)
    fp = np.zeros(len(preds_all), dtype=np.float32)

    for i, (img_id, sc, poly_p) in enumerate(preds_all):
        gt_polys = gts.get(img_id, [])
        if not gt_polys:
            fp[i] = 1.0
            continue
        best_iou = 0.0
        best_j = -1
        for j, poly_g in enumerate(gt_polys):
            if used[img_id][j]:
                continue
            iou = _poly_iou(poly_p, poly_g)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= float(args.iou) and best_j >= 0:
            tp[i] = 1.0
            used[img_id][best_j] = True
        else:
            fp[i] = 1.0

    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    rec = tp_c / max(float(npos), 1.0)
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-6)
    ap_val = _ap_from_pr(rec, prec)
    print(
        json.dumps(
            {
                "iou": float(args.iou),
                "min_score": float(args.min_score),
                "num_gt": int(npos),
                "num_preds": int(len(preds_all)),
                "AP": float(ap_val),
                "final_precision": float(prec[-1]) if prec.size else 0.0,
                "final_recall": float(rec[-1]) if rec.size else 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
