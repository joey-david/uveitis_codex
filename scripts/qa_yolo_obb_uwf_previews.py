#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def _poly_area(poly: np.ndarray) -> float:
    return float(abs(cv2.contourArea(poly.astype(np.float32))))


def _poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    inter_area, _ = cv2.intersectConvexConvex(a, b)
    if inter_area <= 0:
        return 0.0
    ua = _poly_area(a) + _poly_area(b) - float(inter_area)
    return float(inter_area) / ua if ua > 0 else 0.0


def _nms_polys(polys: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    if polys.shape[0] == 0:
        return []
    if iou <= 0:
        return list(range(polys.shape[0]))
    order = np.argsort(-scores)
    keep: list[int] = []
    for idx in order.tolist():
        ok = True
        for j in keep:
            if _poly_iou(polys[idx], polys[j]) > iou:
                ok = False
                break
        if ok:
            keep.append(idx)
    return keep


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_image_ids(split_json: Path, split: str, n: int) -> list[str]:
    s = _read_json(split_json)
    ids = [x for x in s.get(split, []) if str(x).startswith("uwf700::")]
    return ids[:n]

def _pick_image_ids_from_coco(coco_json: Path, n: int) -> list[str]:
    coco = _read_json(coco_json)
    ids = []
    for im in coco.get("images", []):
        if im.get("image_id"):
            ids.append(str(im["image_id"]))
    # keep stable order, unique
    seen = set()
    out = []
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out[:n]


def _load_names(class_map_yaml: Path) -> list[str]:
    cfg = yaml.safe_load(class_map_yaml.read_text(encoding="utf-8"))
    return [str(x) for x in cfg["categories"]]

def _load_names_from_data_yaml(data_yaml: Path) -> list[str]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    if isinstance(names, list):
        return [str(x) for x in names]
    raise SystemExit(f"Unexpected `names` in {data_yaml}")


def _draw_poly(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    cv2.polylines(img, [pts.astype(np.int32)], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _label_color(cls: int) -> tuple[int, int, int]:
    # BGR: stable distinct-ish colors.
    colors = [
        (0, 255, 255),
        (0, 200, 0),
        (255, 200, 0),
        (255, 0, 255),
        (0, 128, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 255, 128),
    ]
    return colors[int(cls) % len(colors)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a few UWF preview overlays using a YOLOv8-OBB model trained on tiles.")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--coco", type=Path, default=None, help="Optional: pick image_ids from this COCO json instead of split json.")
    ap.add_argument("--split-json", type=Path, default=Path("splits/stage0_0.json"))
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--tile-iou", type=float, default=0.7, help="YOLO per-tile NMS IoU.")
    ap.add_argument("--max-det", type=int, default=200)
    ap.add_argument("--global-nms-iou", type=float, default=0.3, help="Global polygon NMS IoU across tiles (<=0 disables).")
    ap.add_argument("--class-map", type=Path, default=Path("configs/class_map.yaml"))
    ap.add_argument("--data-yaml", type=Path, default=None, help="If set, read class names from this YOLO data.yaml.")
    ap.add_argument("--out-dir", type=Path, default=Path("eval/yolo_obb_previews"))
    args = ap.parse_args()

    names = _load_names_from_data_yaml(args.data_yaml) if args.data_yaml else _load_names(args.class_map)
    model = YOLO(args.weights.as_posix())

    out_dir = args.out_dir / f"{args.weights.stem}_{args.split}_n{args.n}_conf{args.conf}"
    _ensure_dir(out_dir)

    if args.coco:
        image_ids = _pick_image_ids_from_coco(args.coco, args.n * 3)
    else:
        image_ids = _pick_image_ids(args.split_json, args.split, args.n * 3)
    kept = 0
    for image_id in image_ids:
        img_id = image_id.replace("::", "__")
        global_path = Path("preproc/global_1024") / f"{img_id}.png"
        meta_path = Path("preproc/tiles_meta") / f"{img_id}.json"
        if not global_path.exists() or not meta_path.exists():
            continue

        meta = _read_json(meta_path)
        global_bgr = cv2.imread(str(global_path), cv2.IMREAD_COLOR)
        if global_bgr is None:
            continue

        # Predict per-tile and project to global by adding tile offsets, then global NMS.
        all_xy = []
        all_cls = []
        all_conf = []
        for t in meta["tiles"]:
            tile_id = t["tile_id"]
            x0, y0, x1, y1 = int(t["x0"]), int(t["y0"]), int(t["x1"]), int(t["y1"])
            tile_bgr = global_bgr[y0:y1, x0:x1]
            res = model.predict(
                source=tile_bgr,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.tile_iou,
                device=0,
                verbose=False,
                max_det=args.max_det,
            )
            r = res[0]
            if r.obb is None or len(r.obb) == 0:
                continue

            xy = r.obb.xyxyxyxy.cpu().numpy()  # (N,4,2) in tile pixels
            cls = r.obb.cls.cpu().numpy().astype(int)
            conf = r.obb.conf.cpu().numpy()

            for i in range(xy.shape[0]):
                pts = xy[i].copy()
                pts[:, 0] += float(x0)
                pts[:, 1] += float(y0)
                all_xy.append(pts)
                all_cls.append(cls[i])
                all_conf.append(conf[i])

        if not all_xy:
            continue

        all_xy = np.stack(all_xy, axis=0).astype(np.float32)
        all_cls = np.asarray(all_cls, dtype=np.int32)
        all_conf = np.asarray(all_conf, dtype=np.float32)

        keep = []
        for c in sorted(set(all_cls.tolist())):
            idx = np.where(all_cls == c)[0]
            k = _nms_polys(all_xy[idx], all_conf[idx], args.global_nms_iou)
            keep.extend(idx[k].tolist())
        keep = sorted(keep, key=lambda i: float(all_conf[i]), reverse=True)

        for i in keep:
            pts = all_xy[i]
            c = _label_color(int(all_cls[i]))
            _draw_poly(global_bgr, pts, c, thickness=2)
            label = names[int(all_cls[i])] if 0 <= int(all_cls[i]) < len(names) else str(int(all_cls[i]))
            p0 = pts[0]
            cv2.putText(
                global_bgr,
                f"{label} {float(all_conf[i]):.2f}",
                (int(p0[0]), int(max(12, p0[1]))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                c,
                2,
                lineType=cv2.LINE_AA,
            )

        out_path = out_dir / f"{img_id}__preds{len(keep)}_raw{all_xy.shape[0]}.png"
        cv2.imwrite(str(out_path), global_bgr)
        kept += 1
        if kept >= args.n:
            break

    print(f"Wrote {kept} overlays to {out_dir}")


if __name__ == "__main__":
    main()
