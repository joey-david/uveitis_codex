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


def _pick_image_ids_from_coco(coco_json: Path, n: int) -> list[str]:
    coco = _read_json(coco_json)
    ids = []
    for im in coco.get("images", []):
        if im.get("image_id"):
            ids.append(str(im["image_id"]))
    seen = set()
    out = []
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out[:n]


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


def _pred_tile(model: YOLO, tile_bgr: np.ndarray, imgsz: int, conf: float, iou: float, max_det: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    res = model.predict(source=tile_bgr, imgsz=imgsz, conf=conf, iou=iou, device=0, verbose=False, max_det=max_det)
    r = res[0]
    if r.obb is None or len(r.obb) == 0:
        return np.zeros((0, 4, 2), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)
    return (
        r.obb.xyxyxyxy.cpu().numpy(),
        r.obb.cls.cpu().numpy().astype(int),
        r.obb.conf.cpu().numpy(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay predictions from main + vascularite YOLOv8-OBB models onto UWF globals.")
    ap.add_argument("--main-weights", type=Path, required=True)
    ap.add_argument("--main-data", type=Path, required=True)
    ap.add_argument("--vasc-weights", type=Path, required=True)
    ap.add_argument("--vasc-data", type=Path, required=True)
    ap.add_argument("--coco", type=Path, default=Path("labels_coco/uwf700_val_tiles.json"))
    ap.add_argument(
        "--image-id",
        action="append",
        default=[],
        help="Optional explicit image_id(s) (repeatable). If set, overrides --coco selection.",
    )
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf-main", type=float, default=0.05)
    ap.add_argument("--conf-vasc", type=float, default=0.05)
    ap.add_argument("--tile-iou", type=float, default=0.7, help="YOLO per-tile NMS IoU.")
    ap.add_argument("--max-det", type=int, default=200)
    ap.add_argument("--global-nms-iou", type=float, default=0.3, help="Global polygon NMS IoU across tiles (<=0 disables).")
    ap.add_argument("--out-dir", type=Path, default=Path("eval/yolo_obb_previews_combined"))
    args = ap.parse_args()

    main_names = _load_names_from_data_yaml(args.main_data)
    vasc_names = _load_names_from_data_yaml(args.vasc_data)
    main_model = YOLO(args.main_weights.as_posix())
    vasc_model = YOLO(args.vasc_weights.as_posix())

    out_dir = args.out_dir / f"{args.main_weights.stem}__plus__{args.vasc_weights.stem}__n{args.n}"
    _ensure_dir(out_dir)

    image_ids = [str(x) for x in (args.image_id or [])] or _pick_image_ids_from_coco(args.coco, args.n * 3)
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

        main_xy = []
        main_cls = []
        main_conf = []
        vasc_xy = []
        vasc_cls = []
        vasc_conf = []
        for t in meta["tiles"]:
            x0, y0, x1, y1 = int(t["x0"]), int(t["y0"]), int(t["x1"]), int(t["y1"])
            tile_bgr = global_bgr[y0:y1, x0:x1]

            xy, cls, conf = _pred_tile(main_model, tile_bgr, args.imgsz, args.conf_main, args.tile_iou, args.max_det)
            for i in range(xy.shape[0]):
                pts = xy[i].copy()
                pts[:, 0] += float(x0)
                pts[:, 1] += float(y0)
                main_xy.append(pts)
                main_cls.append(cls[i])
                main_conf.append(conf[i])

            xy, cls, conf = _pred_tile(vasc_model, tile_bgr, args.imgsz, args.conf_vasc, args.tile_iou, args.max_det)
            for i in range(xy.shape[0]):
                pts = xy[i].copy()
                pts[:, 0] += float(x0)
                pts[:, 1] += float(y0)
                vasc_xy.append(pts)
                vasc_cls.append(cls[i])
                vasc_conf.append(conf[i])

        total_main = len(main_xy)
        total_vasc = len(vasc_xy)
        kept_main = 0
        kept_vasc = 0
        if main_xy:
            main_xy = np.stack(main_xy, axis=0).astype(np.float32)
            main_cls = np.asarray(main_cls, dtype=np.int32)
            main_conf = np.asarray(main_conf, dtype=np.float32)
            keep = []
            for c in sorted(set(main_cls.tolist())):
                idx = np.where(main_cls == c)[0]
                k = _nms_polys(main_xy[idx], main_conf[idx], args.global_nms_iou)
                keep.extend(idx[k].tolist())
            keep = sorted(keep, key=lambda i: float(main_conf[i]), reverse=True)
            kept_main = len(keep)
            for i in keep:
                pts = main_xy[i]
                c = _label_color(int(main_cls[i]))
                _draw_poly(global_bgr, pts, c, thickness=2)
                label = main_names[int(main_cls[i])] if 0 <= int(main_cls[i]) < len(main_names) else str(int(main_cls[i]))
                p0 = pts[0]
                cv2.putText(
                    global_bgr,
                    f"{label} {float(main_conf[i]):.2f}",
                    (int(p0[0]), int(max(12, p0[1]))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    c,
                    2,
                    lineType=cv2.LINE_AA,
                )

        if vasc_xy:
            vasc_xy = np.stack(vasc_xy, axis=0).astype(np.float32)
            vasc_cls = np.asarray(vasc_cls, dtype=np.int32)
            vasc_conf = np.asarray(vasc_conf, dtype=np.float32)
            keep = []
            for c in sorted(set(vasc_cls.tolist())):
                idx = np.where(vasc_cls == c)[0]
                k = _nms_polys(vasc_xy[idx], vasc_conf[idx], args.global_nms_iou)
                keep.extend(idx[k].tolist())
            keep = sorted(keep, key=lambda i: float(vasc_conf[i]), reverse=True)
            kept_vasc = len(keep)
            for i in keep:
                pts = vasc_xy[i]
                c = (0, 0, 255)  # red
                _draw_poly(global_bgr, pts, c, thickness=3)
                label = vasc_names[int(vasc_cls[i])] if 0 <= int(vasc_cls[i]) < len(vasc_names) else "vascularite"
                p0 = pts[0]
                cv2.putText(
                    global_bgr,
                    f"{label} {float(vasc_conf[i]):.2f}",
                    (int(p0[0]), int(max(12, p0[1] + 14))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    c,
                    2,
                    lineType=cv2.LINE_AA,
                )

        out_path = out_dir / f"{img_id}__main{kept_main}_raw{total_main}__vasc{kept_vasc}_raw{total_vasc}.png"
        cv2.imwrite(str(out_path), global_bgr)
        kept += 1
        if kept >= args.n:
            break

    print(f"Wrote {kept} overlays to {out_dir}")


if __name__ == "__main__":
    main()
