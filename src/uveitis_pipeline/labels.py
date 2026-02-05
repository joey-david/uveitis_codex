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
        points = [float(v) for v in parts[1:]]
        xs = [points[i] * width for i in range(0, len(points), 2)]
        ys = [points[i] * height for i in range(1, len(points), 2)]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        anns.append(
            {
                "class_name": class_map.get(cls_raw, f"uwf_{cls_raw}"),
                "bbox_xyxy": [x1, y1, x2, y2],
                "obb": points,
            }
        )
    return anns


def _parse_fgadr_masks(image_name: str, root: Path, class_map: dict[str, str], min_area: int) -> list[dict]:
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
        for box in _connected_component_boxes(mask, min_area=min_area):
            anns.append({"class_name": cls_name, "bbox_xyxy": box})
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
            anns_src = _parse_fgadr_masks(Path(row["filepath"]).name, Path(row["labels_path"]), fgadr_map, min_comp_area)

        global_anns: list[dict] = []
        for ann in anns_src:
            if ann["class_name"] not in cat_to_id:
                continue
            gbox = _project_to_global(ann["bbox_xyxy"], crop_meta, global_meta)
            gbox = _clip_box(gbox, g_w, g_h)
            if gbox is None:
                continue
            global_anns.append({"class_id": cat_to_id[ann["class_name"]], "bbox": gbox, "obb": ann.get("obb")})

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
                dbg = draw_boxes(read_image(global_path), boxes, labels)
                write_image(out_debug_dir / f"{image_id}_overlay.png", dbg)
            continue

        for tile_meta in tiles_meta["tiles"]:
            tile_id = tile_meta["tile_id"]
            tile_box = [tile_meta["x0"], tile_meta["y0"], tile_meta["x1"], tile_meta["y1"]]
            tile_file = preproc_root / "tiles" / image_id / f"{tile_id}.png"
            if not tile_file.exists():
                continue

            anns_tile = []
            for ann in global_anns:
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
