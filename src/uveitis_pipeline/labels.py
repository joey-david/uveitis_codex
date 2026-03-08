from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .common import draw_boxes, ensure_dir, read_image, save_json, write_image


def _connected_component_boxes(mask: np.ndarray, min_area: int) -> list[list[float]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes: list[list[float]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return boxes


def _poly_bbox_xyxy(poly: np.ndarray) -> list[float] | None:
    if poly.size == 0:
        return None
    xs = poly[:, 0]
    ys = poly[:, 1]
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _poly_to_obb(poly: np.ndarray) -> list[float] | None:
    if poly.shape[0] < 3:
        return None
    rect = cv2.minAreaRect(poly.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float32)
    return [float(v) for xy in box.tolist() for v in xy]


def _clip_obb_to_tile(obb_norm: list[float], tile_box: list[float], g_w: int, g_h: int) -> dict | None:
    if len(obb_norm) != 8:
        return None
    pts = np.array(list(zip(obb_norm[0::2], obb_norm[1::2])), dtype=np.float32)
    pts[:, 0] *= float(g_w)
    pts[:, 1] *= float(g_h)

    x0, y0, x1, y1 = [float(v) for v in tile_box]
    tile_poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    src_area = float(abs(cv2.contourArea(pts)))
    if src_area <= 1.0:
        return None

    inter_area, inter_poly = cv2.intersectConvexConvex(pts, tile_poly)
    if inter_poly is None or float(inter_area) <= 1.0:
        return None
    inter = inter_poly.reshape(-1, 2).astype(np.float32)

    bbox_g = _poly_bbox_xyxy(inter)
    obb_g = _poly_to_obb(inter)
    if bbox_g is None or obb_g is None:
        return None

    tw = max(1.0, x1 - x0)
    th = max(1.0, y1 - y0)
    bbox_t = [
        float(np.clip(bbox_g[0] - x0, 0, tw - 1)),
        float(np.clip(bbox_g[1] - y0, 0, th - 1)),
        float(np.clip(bbox_g[2] - x0, 0, tw - 1)),
        float(np.clip(bbox_g[3] - y0, 0, th - 1)),
    ]
    if bbox_t[2] <= bbox_t[0] or bbox_t[3] <= bbox_t[1]:
        return None

    obb_t = []
    for i in range(0, 8, 2):
        tx = float(np.clip((obb_g[i] - x0) / tw, 0.0, 1.0))
        ty = float(np.clip((obb_g[i + 1] - y0) / th, 0.0, 1.0))
        obb_t.extend([tx, ty])

    return {
        "bbox": bbox_t,
        "obb": obb_t,
        "area_ratio": float(inter_area) / src_area,
    }


def _parse_uwf_obb(
    label_path: Path,
    width: int,
    height: int,
    class_map: dict[str, str],
    image_path: Path | None = None,
) -> list[dict]:
    anns = []
    if not label_path.exists():
        return anns
    if (width <= 0 or height <= 0) and image_path is not None and image_path.exists():
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is not None:
            height, width = img.shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        cls_raw = str(int(float(parts[0])))
        points = [float(v) for v in parts[1:]]  # normalized x1 y1 ... x4 y4 in raw image coords
        xs = [points[i] * width for i in range(0, len(points), 2)]
        ys = [points[i] * height for i in range(1, len(points), 2)]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        anns.append(
            {
                "class_name": class_map.get(cls_raw, f"uwf_{cls_raw}"),
                "bbox_xyxy": [x1, y1, x2, y2],
                "obb": points,
                "obb_xy": [v for xy in zip(xs, ys) for v in xy],
            }
        )
    return anns


def _parse_fgadr_masks(
    image_name: str,
    root: Path,
    class_map: dict[str, str],
    min_area: int,
    mask_to_obb: bool = True,
) -> list[dict]:
    anns = []
    for mask_dir in sorted(root.glob("*_Masks")):
        mask_path = mask_dir / image_name
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8)
        cls_name = class_map.get(mask_dir.name, mask_dir.name.replace("_Masks", "").lower())
        if not mask_to_obb:
            for box in _connected_component_boxes(mask, min_area=min_area):
                anns.append({"class_name": cls_name, "bbox_xyxy": box})
            continue

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            _, _, _, _, area = stats[i]
            if int(area) < min_area:
                continue
            comp = (labels == i).astype(np.uint8)
            res = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = res[0] if len(res) == 2 else res[1]
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            if float(cv2.contourArea(cnt)) < float(min_area):
                continue
            poly = cnt.reshape(-1, 2).astype(np.float32)
            bbox = _poly_bbox_xyxy(poly)
            obb_xy = _poly_to_obb(poly)
            if bbox is None or obb_xy is None:
                continue
            anns.append({"class_name": cls_name, "bbox_xyxy": bbox, "obb_xy": obb_xy})
    return anns


def _project_to_global(box: list[float], crop_meta: dict, global_meta: dict) -> list[float]:
    cx0, cy0, _, _ = crop_meta["bbox_xyxy"]
    px, py = global_meta["pad_xy"]
    scale = global_meta["scale"]

    x1 = (box[0] - cx0 + px) * scale
    y1 = (box[1] - cy0 + py) * scale
    x2 = (box[2] - cx0 + px) * scale
    y2 = (box[3] - cy0 + py) * scale
    return [x1, y1, x2, y2]


def _project_point_to_global(x: float, y: float, crop_meta: dict, global_meta: dict) -> tuple[float, float]:
    cx0, cy0, _, _ = crop_meta["bbox_xyxy"]
    px, py = global_meta["pad_xy"]
    scale = global_meta["scale"]
    return ((x - cx0 + px) * scale, (y - cy0 + py) * scale)


def _clip_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    x = float(np.clip(x, 0, width - 1))
    y = float(np.clip(y, 0, height - 1))
    return x, y


def _clip_box(box: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, width - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _to_coco_bbox(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def _intersect(a: list[float], b: list[float]) -> list[float] | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def build_coco_from_manifest(
    manifest_rows: list[dict],
    split_ids: set[str],
    class_map_cfg: dict,
    preproc_root: Path,
    out_coco_path: Path,
    out_debug_dir: Path,
    tile_mode: bool,
    min_comp_area: int = 8,
    min_tile_box_ratio: float = 0.2,
    fgadr_mask_to_obb: bool = True,
    tile_require_obb_for_obb_sources: bool = True,
    debug_max_images: int = 0,
) -> dict:
    categories = class_map_cfg["categories"]
    cat_to_id = {c: i + 1 for i, c in enumerate(categories)}
    uwf_map = {str(k): v for k, v in class_map_cfg["maps"].get("uwf700", {}).items()}
    fgadr_map = {str(k): v for k, v in class_map_cfg["maps"].get("fgadr", {}).items()}

    out_debug_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": c} for i, c in enumerate(categories)],
    }
    ann_id = 1
    dbg_written = 0

    for row in manifest_rows:
        if row["image_id"] not in split_ids:
            continue
        image_id = row["image_id"].replace("::", "__")

        crop_meta_path = preproc_root / "crop_meta" / f"{image_id}.json"
        tiles_meta_path = preproc_root / "tiles_meta" / f"{image_id}.json"
        norm_path = preproc_root / "norm" / f"{image_id}.png"
        global_path = preproc_root / "global_1024" / f"{image_id}.png"

        if not crop_meta_path.exists() or not tiles_meta_path.exists() or not global_path.exists():
            continue

        import json

        crop_meta = json.loads(crop_meta_path.read_text(encoding="utf-8"))
        tiles_meta = json.loads(tiles_meta_path.read_text(encoding="utf-8"))
        global_meta = tiles_meta["global_meta"]
        g_w, g_h = tiles_meta["global_size"]

        anns_src: list[dict] = []
        if row["dataset"] == "uwf700" and row["labels_path"]:
            anns_src = _parse_uwf_obb(
                Path(row["labels_path"]),
                int(row.get("width", 0) or 0),
                int(row.get("height", 0) or 0),
                uwf_map,
                Path(row["filepath"]),
            )
        elif row["dataset"] == "fgadr":
            anns_src = _parse_fgadr_masks(
                Path(row["filepath"]).name,
                Path(row["labels_path"]),
                fgadr_map,
                min_comp_area,
                mask_to_obb=fgadr_mask_to_obb,
            )

        global_anns: list[dict] = []
        for ann in anns_src:
            if ann["class_name"] not in cat_to_id:
                continue
            obb_xy = ann.get("obb_xy")
            if obb_xy and len(obb_xy) == 8:
                gpts = []
                for i in range(0, 8, 2):
                    gx, gy = _project_point_to_global(float(obb_xy[i]), float(obb_xy[i + 1]), crop_meta, global_meta)
                    gx, gy = _clip_point(gx, gy, g_w, g_h)
                    gpts.append((gx, gy))
                xs = [p[0] for p in gpts]
                ys = [p[1] for p in gpts]
                gbox = _clip_box([min(xs), min(ys), max(xs), max(ys)], g_w, g_h)
                if gbox is None:
                    continue
                obb_norm = [v for x, y in gpts for v in (x / g_w, y / g_h)]
                global_anns.append({"class_id": cat_to_id[ann["class_name"]], "bbox": gbox, "obb": obb_norm})
            else:
                gbox = _project_to_global(ann["bbox_xyxy"], crop_meta, global_meta)
                gbox = _clip_box(gbox, g_w, g_h)
                if gbox is None:
                    continue
                global_anns.append({"class_id": cat_to_id[ann["class_name"]], "bbox": gbox})

        if not tile_mode:
            coco_img_id = len(coco["images"]) + 1
            coco["images"].append(
                {
                    "id": coco_img_id,
                    "file_name": global_path.as_posix(),
                    "width": g_w,
                    "height": g_h,
                    "image_id": row["image_id"],
                }
            )
            boxes = []
            labels = []
            for ann in global_anns:
                coco_box = _to_coco_bbox(ann["bbox"])
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": coco_img_id,
                        "category_id": ann["class_id"],
                        "bbox": coco_box,
                        "area": coco_box[2] * coco_box[3],
                        "iscrowd": 0,
                        "obb": ann.get("obb"),
                    }
                )
                ann_id += 1
                boxes.append(ann["bbox"])
                labels.append(categories[ann["class_id"] - 1])
            if boxes:
                if debug_max_images > 0 and dbg_written < debug_max_images:
                    dbg = draw_boxes(read_image(global_path), boxes, labels)
                    write_image(out_debug_dir / f"{image_id}_overlay.png", dbg)
                    dbg_written += 1
            continue

        for tile_meta in tiles_meta["tiles"]:
            tile_id = tile_meta["tile_id"]
            tile_box = [tile_meta["x0"], tile_meta["y0"], tile_meta["x1"], tile_meta["y1"]]
            tile_file = preproc_root / "tiles" / image_id / f"{tile_id}.png"
            if not tile_file.exists():
                continue

            anns_tile = []
            for ann in global_anns:
                if ann.get("obb") and len(ann["obb"]) == 8:
                    clipped = _clip_obb_to_tile(ann["obb"], tile_box, g_w, g_h)
                    if clipped is None:
                        if tile_require_obb_for_obb_sources:
                            continue
                        inter = _intersect(ann["bbox"], tile_box)
                        if inter is None:
                            continue
                        clipped = {
                            "bbox": [
                                inter[0] - tile_box[0],
                                inter[1] - tile_box[1],
                                inter[2] - tile_box[0],
                                inter[3] - tile_box[1],
                            ],
                            "obb": None,
                            "area_ratio": _box_area(inter) / max(_box_area(ann["bbox"]), 1.0),
                        }
                    if clipped["area_ratio"] < min_tile_box_ratio:
                        continue
                    tile_ann = {
                        "class_id": ann["class_id"],
                        "bbox": clipped["bbox"],
                    }
                    if clipped.get("obb") and len(clipped["obb"]) == 8:
                        tile_ann["obb"] = clipped["obb"]
                    anns_tile.append(tile_ann)
                    continue

                inter = _intersect(ann["bbox"], tile_box)
                if inter is None:
                    continue
                if _box_area(inter) / max(_box_area(ann["bbox"]), 1.0) < min_tile_box_ratio:
                    continue
                anns_tile.append(
                    {
                        "class_id": ann["class_id"],
                        "bbox": [
                            inter[0] - tile_box[0],
                            inter[1] - tile_box[1],
                            inter[2] - tile_box[0],
                            inter[3] - tile_box[1],
                        ],
                    }
                )

            coco_img_id = len(coco["images"]) + 1
            coco["images"].append(
                {
                    "id": coco_img_id,
                    "file_name": tile_file.as_posix(),
                    "width": int(tile_box[2] - tile_box[0]),
                    "height": int(tile_box[3] - tile_box[1]),
                    "image_id": row["image_id"],
                    "tile_id": tile_id,
                }
            )

            for ann in anns_tile:
                coco_box = _to_coco_bbox(ann["bbox"])
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": coco_img_id,
                        "category_id": ann["class_id"],
                        "bbox": coco_box,
                        "area": coco_box[2] * coco_box[3],
                        "iscrowd": 0,
                        "obb": ann.get("obb"),
                    }
                )
                ann_id += 1

    save_json(out_coco_path, coco)
    return coco


def summarize_coco(coco: dict) -> dict:
    class_counts = defaultdict(int)
    areas = []
    for ann in coco["annotations"]:
        class_counts[ann["category_id"]] += 1
        areas.append(float(ann["area"]))
    return {
        "num_images": len(coco["images"]),
        "num_annotations": len(coco["annotations"]),
        "class_counts": dict(class_counts),
        "avg_box_area": float(np.mean(areas)) if areas else 0.0,
    }
