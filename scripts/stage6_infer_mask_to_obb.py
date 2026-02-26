#!/usr/bin/env python3
"""Infer masks with RETFound mask model, then convert to OBB/polygon predictions."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

import cv2
import numpy as np
import torch

from uveitis_pipeline.common import ensure_dir, load_yaml, read_jsonl, save_jsonl, write_image
from uveitis_pipeline.retfound_mask import RetFoundEncoder, RetFoundMaskModel, load_retfound_vit, load_retfound_weights


def _xyxy_iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two axis-aligned xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(ua + ub - inter, 1e-6)


def _nms(preds: list[dict], iou_thr: float) -> list[dict]:
    """Per-class NMS on axis-aligned boxes."""
    out: list[dict] = []
    by_class: dict[int, list[dict]] = {}
    for p in preds:
        by_class.setdefault(int(p["class_id"]), []).append(p)

    for _, items in by_class.items():
        items = sorted(items, key=lambda x: float(x["score"]), reverse=True)
        keep: list[dict] = []
        while items:
            cur = items.pop(0)
            keep.append(cur)
            nxt = []
            for cand in items:
                if _xyxy_iou(cur["bbox_xyxy"], cand["bbox_xyxy"]) <= iou_thr:
                    nxt.append(cand)
            items = nxt
        out.extend(keep)
    return out


def _cap_per_class(preds: list[dict], max_per_class: int) -> list[dict]:
    """Keep at most K predictions per class by score."""
    k = int(max_per_class)
    if k <= 0:
        return preds
    out: list[dict] = []
    by_class: dict[int, list[dict]] = {}
    for p in preds:
        by_class.setdefault(int(p["class_id"]), []).append(p)
    for _, items in by_class.items():
        items = sorted(items, key=lambda x: float(x["score"]), reverse=True)[:k]
        out.extend(items)
    return out


def _extract_components(prob: np.ndarray, cls_id: int, cls_name: str, cfg: dict) -> list[dict]:
    """Extract contour components from one class probability map."""
    thr = float(cfg.get("threshold", 0.5))
    min_area = int(cfg.get("min_area_px", 16))
    simplify_eps = float(cfg.get("polygon_simplify_eps", 1.25))
    open_k = int(cfg.get("open_kernel", 0))
    close_k = int(cfg.get("close_kernel", 3))

    mask = (prob >= thr).astype(np.uint8)
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    h, w = mask.shape
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    out: list[dict] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(min_area):
            continue
        poly = cnt.reshape(-1, 2).astype(np.float32)
        if simplify_eps > 0:
            poly = cv2.approxPolyDP(poly, epsilon=simplify_eps, closed=True).reshape(-1, 2).astype(np.float32)
        if poly.shape[0] < 3:
            continue

        comp = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(comp, [np.round(poly).astype(np.int32)], 1)
        score = float(prob[comp > 0].mean()) if np.any(comp > 0) else float(prob[int(poly[0, 1]), int(poly[0, 0])])

        x1 = float(np.clip(poly[:, 0].min(), 0, w - 1))
        y1 = float(np.clip(poly[:, 1].min(), 0, h - 1))
        x2 = float(np.clip(poly[:, 0].max(), 0, w - 1))
        y2 = float(np.clip(poly[:, 1].max(), 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        rect = cv2.minAreaRect(poly)
        obb = cv2.boxPoints(rect).astype(np.float32)

        out.append(
            {
                "class_id": int(cls_id),
                "class_name": cls_name,
                "score": score,
                "bbox_xyxy": [x1, y1, x2, y2],
                "polygon": [float(np.clip(v / (w if i % 2 == 0 else h), 0.0, 1.0)) for i, v in enumerate([vv for xy in poly.tolist() for vv in xy])],
                "obb": [
                    float(np.clip(v / (w if i % 2 == 0 else h), 0.0, 1.0))
                    for i, v in enumerate([vv for xy in obb.tolist() for vv in xy])
                ],
            }
        )

    return out


def _extract_union_components(
    prob_stack: np.ndarray,
    class_names: list[str],
    class_thresholds: dict[str, float],
    cfg: dict,
    class_post_cfg: dict[str, dict],
) -> list[dict]:
    """Extract class-agnostic components and assign class by per-component mean probability."""
    thr = float(cfg.get("threshold", 0.5))
    min_area = int(cfg.get("min_area_px", 16))
    simplify_eps = float(cfg.get("polygon_simplify_eps", 1.25))
    open_k = int(cfg.get("open_kernel", 0))
    close_k = int(cfg.get("close_kernel", 3))

    max_prob = prob_stack.max(axis=0)
    mask = (max_prob >= thr).astype(np.uint8)
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    h, w = mask.shape
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    out: list[dict] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(min_area):
            continue
        poly = cnt.reshape(-1, 2).astype(np.float32)
        if simplify_eps > 0:
            poly = cv2.approxPolyDP(poly, epsilon=simplify_eps, closed=True).reshape(-1, 2).astype(np.float32)
        if poly.shape[0] < 3:
            continue

        comp = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(comp, [np.round(poly).astype(np.int32)], 1)
        pix = comp > 0
        if not np.any(pix):
            continue

        means = prob_stack[:, pix].mean(axis=1)
        cls_idx = int(np.argmax(means))
        cls_name = class_names[cls_idx]
        cls_id = cls_idx + 1
        score = float(means[cls_idx])

        cls_cfg = class_post_cfg.get(cls_name, {})
        cls_min_area = int(cls_cfg.get("min_area_px", min_area))
        if area < float(cls_min_area):
            continue
        cls_thr = float(cls_cfg.get("threshold", class_thresholds.get(cls_name, thr)))
        if score < cls_thr:
            continue

        x1 = float(np.clip(poly[:, 0].min(), 0, w - 1))
        y1 = float(np.clip(poly[:, 1].min(), 0, h - 1))
        x2 = float(np.clip(poly[:, 0].max(), 0, w - 1))
        y2 = float(np.clip(poly[:, 1].max(), 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        rect = cv2.minAreaRect(poly)
        obb = cv2.boxPoints(rect).astype(np.float32)

        out.append(
            {
                "class_id": int(cls_id),
                "class_name": cls_name,
                "score": score,
                "bbox_xyxy": [x1, y1, x2, y2],
                "polygon": [float(np.clip(v / (w if i % 2 == 0 else h), 0.0, 1.0)) for i, v in enumerate([vv for xy in poly.tolist() for vv in xy])],
                "obb": [
                    float(np.clip(v / (w if i % 2 == 0 else h), 0.0, 1.0))
                    for i, v in enumerate([vv for xy in obb.tolist() for vv in xy])
                ],
            }
        )

    return out


def _draw_preview(image: np.ndarray, preds: list[dict]) -> np.ndarray:
    """Draw predicted polygons and class labels."""
    out = image.copy()
    h, w = out.shape[:2]
    for p in preds:
        poly = np.array(list(zip(p["polygon"][0::2], p["polygon"][1::2])), dtype=np.float32)
        poly[:, 0] *= w
        poly[:, 1] *= h
        poly = np.round(poly).astype(np.int32)
        color = (0, 255, 0)
        cv2.polylines(out, [poly], True, color, 2, lineType=cv2.LINE_AA)
        x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
        txt = f"{p['class_name']}:{p['score']:.2f}"
        cv2.putText(out, txt, (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def main() -> None:
    """Run native mask inference and export OBB-style predictions."""
    parser = argparse.ArgumentParser(description="Stage-6 infer masks and export OBB predictions")
    parser.add_argument("--config", default="configs/stage6_infer_mask_to_obb.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ckpt = torch.load(cfg["model"]["checkpoint"], map_location="cpu")
    class_names = ckpt.get("class_names") or load_yaml(cfg["model"]["class_map_active"]).get("categories", [])
    num_classes = len(class_names)
    class_thresholds = {name: float(cfg.get("postprocess", {}).get("threshold", 0.5)) for name in class_names}
    class_post_cfg: dict[str, dict] = {}
    t_json = cfg.get("postprocess", {}).get("thresholds_json")
    if t_json and Path(t_json).exists():
        t_data = json.loads(Path(t_json).read_text(encoding="utf-8"))
        for name, row in t_data.get("classes", {}).items():
            if name in class_thresholds and "best_threshold" in row:
                class_thresholds[name] = float(row["best_threshold"])
    p_json = cfg.get("postprocess", {}).get("class_postprocess_json")
    if p_json and Path(p_json).exists():
        p_data = json.loads(Path(p_json).read_text(encoding="utf-8"))
        class_post_cfg = {str(k): dict(v) for k, v in p_data.get("classes", {}).items() if isinstance(v, dict)}

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

    rows = read_jsonl(cfg["data"]["index_jsonl"])
    max_images = int(cfg["data"].get("max_images", 0))
    if max_images > 0:
        rows = rows[:max_images]

    out_root = ensure_dir(Path(cfg["output"]["out_dir"]))
    previews = ensure_dir(out_root / "previews")
    pred_rows = []

    for i, row in enumerate(rows):
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]

        tta_scales = [float(s) for s in cfg.get("model", {}).get("tta_scales", [1.0])]
        tta_hflip = bool(cfg.get("model", {}).get("tta_hflip", False))
        prob_acc = np.zeros((num_classes, h0, w0), dtype=np.float32)
        tta_n = 0
        with torch.no_grad():
            for scale in tta_scales:
                infer_size = max(64, int(round((image_size * scale) / 16.0) * 16))
                if hasattr(model.encoder.vit.patch_embed, "img_size"):
                    model.encoder.vit.patch_embed.img_size = (infer_size, infer_size)
                inp = cv2.resize(image, (infer_size, infer_size), interpolation=cv2.INTER_AREA)
                ten = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
                prob = torch.sigmoid(model(ten))[0].detach().cpu().numpy()
                for c in range(num_classes):
                    prob_acc[c] += cv2.resize(prob[c], (w0, h0), interpolation=cv2.INTER_LINEAR)
                tta_n += 1

                if tta_hflip:
                    tenf = torch.flip(ten, dims=[3])
                    probf = torch.sigmoid(model(tenf))[0].detach().cpu().numpy()[:, :, ::-1]
                    for c in range(num_classes):
                        prob_acc[c] += cv2.resize(probf[c], (w0, h0), interpolation=cv2.INTER_LINEAR)
                    tta_n += 1

        prob_agg = prob_acc / max(1, tta_n)

        mode = str(cfg.get("postprocess", {}).get("component_mode", "per_class"))
        if mode == "union":
            preds = _extract_union_components(
                prob_stack=prob_agg,
                class_names=class_names,
                class_thresholds=class_thresholds,
                cfg=dict(cfg.get("postprocess", {})),
                class_post_cfg=class_post_cfg,
            )
        else:
            preds = []
            for c, cls_name in enumerate(class_names, start=1):
                cls_prob = prob_agg[c - 1]
                post_cfg = dict(cfg.get("postprocess", {}))
                cls_cfg = class_post_cfg.get(cls_name, {})
                post_cfg.update(cls_cfg)
                if "threshold" not in cls_cfg:
                    post_cfg["threshold"] = float(class_thresholds.get(cls_name, post_cfg.get("threshold", 0.5)))
                cls_preds = _extract_components(
                    prob=cls_prob,
                    cls_id=c,
                    cls_name=cls_name,
                    cfg=post_cfg,
                )
                preds.extend(cls_preds)

        preds = _nms(preds, iou_thr=float(cfg.get("postprocess", {}).get("nms_iou", 0.3)))
        preds = _cap_per_class(preds, max_per_class=int(cfg.get("postprocess", {}).get("max_preds_per_class", 0)))

        pred_rows.append(
            {
                "record_id": row.get("record_id", f"idx_{i}"),
                "image_id": row.get("image_id", ""),
                "image_path": row["image_path"],
                "predictions": preds,
            }
        )

        if i < int(cfg["output"].get("preview_n", 20)):
            vis = _draw_preview(image, preds)
            write_image(previews / f"{i:03d}__{Path(row['image_path']).stem}.png", vis)

    save_jsonl(out_root / "predictions.jsonl", pred_rows)
    print(f"saved: {out_root / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
