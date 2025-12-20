import numpy as np
import cv2
from collections import defaultdict


def obb_to_aabb(obb):
    pts = np.array(obb, dtype=np.float32).reshape(4, 2)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    return float(x_min), float(y_min), float(x_max), float(y_max)


def _poly_area(pts):
    return float(abs(cv2.contourArea(pts)))


def _order_convex(points):
    pts = np.array(points, dtype=np.float32).reshape(-1, 2)
    hull = cv2.convexHull(pts)
    return hull.squeeze(1).astype(np.float32)


def oriented_iou(obb1, obb2):
    try:
        p1 = _order_convex(obb1)
        p2 = _order_convex(obb2)
        inter_area, inter_poly = cv2.intersectConvexConvex(p1, p2)
        if np.isnan(inter_area):
            inter_area = 0.0
        area1 = _poly_area(p1)
        area2 = _poly_area(p2)
        union = area1 + area2 - inter_area
        return float(inter_area / union) if union > 0 else 0.0
    except Exception:
        # Fallback to AABB IoU on error
        return aabb_iou(obb_to_aabb(obb1), obb_to_aabb(obb2))


def aabb_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _ap_from_pr(precisions, recalls):
    if not recalls:
        return 0.0
    r = [0.0] + recalls + [1.0]
    p = [1.0] + precisions + [0.0]
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    ap = 0.0
    for i in range(1, len(r)):
        ap += p[i] * (r[i] - r[i - 1])
    return float(ap)


def _evaluate_class(preds_by_img, gts_by_img, cls, iou_thr=0.5, use_obb=True):
    # Flatten predictions across images
    predictions = []  # list of (img, score, box)
    gt_used = {}
    for img, preds in preds_by_img.items():
        for p in preds:
            if p["cls"] == cls:
                predictions.append((img, float(p.get("score", 0.0)), p["obb"]))
    predictions.sort(key=lambda x: -x[1])
    for img, gts in gts_by_img.items():
        gt_used[img] = [False] * sum(1 for c, _ in gts if c == cls)
    # Prepare GT lookup per image
    gts_by_img_cls = {}
    for img, gts in gts_by_img.items():
        gts_by_img_cls[img] = [(c, obb) for (c, obb) in gts if c == cls]

    tp_flags, fp_flags = [], []
    for img, score, obb_pred in predictions:
        gts_cls = gts_by_img_cls.get(img, [])
        best_iou, best_j = 0.0, -1
        for j, (_, obb_gt) in enumerate(gts_cls):
            if gt_used[img][j]:
                continue
            iou = oriented_iou(obb_pred, obb_gt) if use_obb else aabb_iou(obb_to_aabb(obb_pred), obb_to_aabb(obb_gt))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            gt_used[img][best_j] = True
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    gt_total = sum(len(v) for v in gts_by_img_cls.values())
    precisions, recalls = [], []
    cum_tp, cum_fp = 0, 0
    for tpf, fpf in zip(tp_flags, fp_flags):
        cum_tp += tpf
        cum_fp += fpf
        p = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
        r = cum_tp / gt_total if gt_total > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
    ap = _ap_from_pr(precisions, recalls)
    return {
        "ap": ap,
        "precisions": precisions,
        "recalls": recalls,
        "gt_total": gt_total,
        "tp": sum(tp_flags),
        "fp": sum(fp_flags),
    }


def evaluate_dataset(preds_by_img, gts_by_img, classes, iou_thrs=(0.5, 0.75), use_obb=True):
    per_class = {c: {} for c in classes}
    maps = {thr: [] for thr in iou_thrs}
    for thr in iou_thrs:
        for c in classes:
            res = _evaluate_class(preds_by_img, gts_by_img, c, iou_thr=thr, use_obb=use_obb)
            per_class[c][f"AP@{thr:.2f}"] = res["ap"]
            per_class[c]["GT"] = res["gt_total"]
            maps[thr].append(res["ap"])
    summary = {f"mAP@{thr:.2f}": float(np.nanmean(maps[thr]) if maps[thr] else float("nan")) for thr in iou_thrs}

    # Micro-averaged precision/recall at first threshold
    thr0 = iou_thrs[0]
    # Reuse classify results to compute micro p/r
    total_tp = total_fp = total_gt = 0
    for c in classes:
        res = _evaluate_class(preds_by_img, gts_by_img, c, iou_thr=thr0, use_obb=use_obb)
        total_tp += res["tp"]; total_fp += res["fp"]; total_gt += res["gt_total"]
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / total_gt if total_gt > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0
    summary.update({"micro_prec": micro_prec, "micro_rec": micro_rec, "micro_f1": micro_f1})
    return {"per_class": per_class, **summary}


def evaluate_image(preds, gts, iou_thrs=(0.5, 0.75), use_obb=True):
    """Evaluate a single image; returns dict with metrics for each IoU threshold.
    preds: list of {cls:int, obb:[8], score:float}; gts: list of (cls:int, obb:[8])
    """
    # Build minimal dicts
    preds_by_img = {"img": preds}
    gts_by_img = {"img": gts}
    classes = sorted({p["cls"] for p in preds} | {c for c, _ in gts})
    out = {}
    for thr in iou_thrs:
        # Micro metrics for a single image reduce to standard p/r/f1
        # Use class-specific matching then sum
        total_tp = total_fp = 0
        total_gt = len(gts)
        for c in classes:
            res = _evaluate_class(preds_by_img, gts_by_img, c, iou_thr=thr, use_obb=use_obb)
            total_tp += res["tp"]; total_fp += res["fp"]
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / total_gt if total_gt > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        # AP for single image computed across predictions
        # Aggregate per-class AP mean
        ap_vals = []
        for c in classes:
            res = _evaluate_class(preds_by_img, gts_by_img, c, iou_thr=thr, use_obb=use_obb)
            ap_vals.append(res["ap"]) if res["gt_total"] > 0 else None
        ap_mean = float(np.nanmean(ap_vals)) if ap_vals else float('nan')
        out[thr] = {"prec": prec, "rec": rec, "f1": f1, "ap": ap_mean}
    return out

