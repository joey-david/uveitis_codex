#!/usr/bin/env python3
"""Evaluate YOLO-OBB metrics plus duplicate-prediction index on val images."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def _poly_area(poly: np.ndarray) -> float:
    """Return polygon area."""
    return float(abs(cv2.contourArea(poly.astype(np.float32))))


def _poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return IoU for two convex polygons."""
    ia, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    if ia <= 0:
        return 0.0
    ua = _poly_area(a) + _poly_area(b) - float(ia)
    return float(ia) / max(ua, 1e-9)


def _read_data_yaml(path: Path) -> tuple[Path, Path, Path, list[str]]:
    """Load data.yaml and return val image dir, val labels dir, and class names."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    val_dir = root / cfg["val"]
    val_labels = root / str(cfg["val"]).replace("images", "labels")
    names = cfg.get("names", {})
    if isinstance(names, dict):
        class_names = [str(names[i]) for i in sorted(names)]
    elif isinstance(names, list):
        class_names = [str(x) for x in names]
    else:
        raise ValueError(f"Unexpected names in {path}")
    return root, val_dir, val_labels, class_names


def _read_val_gt_counts(labels_dir: Path) -> dict[int, int]:
    """Count GT instances per class in val labels."""
    counts: dict[int, int] = defaultdict(int)
    for p in sorted(labels_dir.glob("*.txt")):
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            cls = int(ln.split()[0])
            counts[cls] += 1
    return counts


def _duplicate_index(
    model: YOLO,
    val_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    duplicate_iou: float,
    device: str,
) -> dict:
    """Compute duplicate overlap index across val predictions."""
    image_paths = sorted([p for p in val_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    dup_pairs = 0
    total_preds = 0
    total_images = 0
    cls_dup = defaultdict(int)
    cls_cnt = defaultdict(int)
    confs: list[float] = []

    for p in image_paths:
        res = model.predict(
            source=str(p),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )[0]
        if res.obb is None or len(res.obb) == 0:
            total_images += 1
            continue
        polys = res.obb.xyxyxyxy.cpu().numpy().astype(np.float32)
        clss = res.obb.cls.cpu().numpy().astype(int)
        scores = res.obb.conf.cpu().numpy().astype(np.float32)
        total_images += 1
        total_preds += int(polys.shape[0])
        confs.extend([float(s) for s in scores.tolist()])

        for cls in sorted(set(clss.tolist())):
            idx = np.where(clss == cls)[0]
            cls_cnt[int(cls)] += int(len(idx))
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    ov = _poly_iou(polys[idx[i]], polys[idx[j]])
                    if ov >= duplicate_iou:
                        dup_pairs += 1
                        cls_dup[int(cls)] += 1

    dup_per_pred = float(dup_pairs / total_preds) if total_preds > 0 else 0.0
    dup_per_img = float(dup_pairs / total_images) if total_images > 0 else 0.0
    conf_arr = np.asarray(confs, dtype=np.float32) if confs else np.zeros((0,), dtype=np.float32)
    conf_stats = {
        "n": int(conf_arr.size),
        "mean": float(np.mean(conf_arr)) if conf_arr.size else 0.0,
        "p50": float(np.percentile(conf_arr, 50)) if conf_arr.size else 0.0,
        "p90": float(np.percentile(conf_arr, 90)) if conf_arr.size else 0.0,
    }
    return {
        "num_images": total_images,
        "num_predictions": total_preds,
        "duplicate_pairs": dup_pairs,
        "duplicate_pairs_per_prediction": dup_per_pred,
        "duplicate_pairs_per_image": dup_per_img,
        "class_duplicate_pairs": {str(k): int(v) for k, v in sorted(cls_dup.items())},
        "class_prediction_counts": {str(k): int(v) for k, v in sorted(cls_cnt.items())},
        "confidence_stats": conf_stats,
    }


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Evaluate YOLO OBB model + duplicate overlap index.")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--duplicate-iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    _, val_dir, val_labels, class_names = _read_data_yaml(args.data)
    gt_counts = _read_val_gt_counts(val_labels)
    model = YOLO(args.model.as_posix())
    val = model.val(
        data=args.data.as_posix(),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
        project="/tmp",
        name=f"diag_val_{args.model.stem}",
    )
    class_maps = [float(x) for x in getattr(val.box, "maps", [])]
    per_class = {}
    present_class = {}
    for i, apv in enumerate(class_maps):
        name = class_names[i] if i < len(class_names) else str(i)
        per_class[name] = apv
        if gt_counts.get(i, 0) > 0:
            present_class[name] = apv

    dup = _duplicate_index(
        model=model,
        val_dir=val_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        duplicate_iou=args.duplicate_iou,
        device=args.device,
    )
    out = {
        "model": args.model.as_posix(),
        "data": args.data.as_posix(),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "duplicate_iou": args.duplicate_iou,
        "map50": float(val.box.map50),
        "map50_95": float(val.box.map),
        "per_class_map50_95": per_class,
        "present_class_map50_95": present_class,
        "val_gt_counts": {class_names[i]: int(gt_counts.get(i, 0)) for i in range(len(class_names))},
        "duplicate_metrics": dup,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
