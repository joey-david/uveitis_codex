"""Build fine-grained native labels (polygons/masks) from manifests and preprocessing artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .common import ensure_dir, read_image, save_json, save_jsonl


def _poly_bbox_xyxy(poly: np.ndarray) -> list[float] | None:
    """Return [x1,y1,x2,y2] for a polygon in pixel space."""
    if poly.size == 0:
        return None
    x1 = float(poly[:, 0].min())
    y1 = float(poly[:, 1].min())
    x2 = float(poly[:, 0].max())
    y2 = float(poly[:, 1].max())
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _poly_to_obb(poly: np.ndarray) -> list[float] | None:
    """Return 4-point OBB from polygon as flat pixel coordinates."""
    if poly.shape[0] < 3:
        return None
    rect = cv2.minAreaRect(poly.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float32)
    return [float(v) for xy in box.tolist() for v in xy]


def _to_norm_poly(poly: np.ndarray, width: int, height: int) -> list[float]:
    """Normalize polygon points to [0,1] in flat x1,y1,... format."""
    out: list[float] = []
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    for x, y in poly.tolist():
        out.append(float(np.clip(x / w, 0.0, 1.0)))
        out.append(float(np.clip(y / h, 0.0, 1.0)))
    return out


def _to_norm_box_xywh(bbox_xyxy: list[float], width: int, height: int) -> list[float]:
    """Normalize xyxy box to xywh."""
    x1, y1, x2, y2 = bbox_xyxy
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    return [
        float(np.clip(x1 / w, 0.0, 1.0)),
        float(np.clip(y1 / h, 0.0, 1.0)),
        float(np.clip((x2 - x1) / w, 0.0, 1.0)),
        float(np.clip((y2 - y1) / h, 0.0, 1.0)),
    ]


def _contours(mask: np.ndarray) -> list[np.ndarray]:
    """Extract external contours from a binary mask."""
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return res[0] if len(res) == 2 else res[1]


def _project_point_to_global(x: float, y: float, crop_meta: dict, global_meta: dict) -> tuple[float, float]:
    """Project a raw-image point into preprocessed global image coordinates."""
    cx0, cy0, _, _ = crop_meta["bbox_xyxy"]
    px, py = global_meta["pad_xy"]
    scale = global_meta["scale"]
    return ((x - cx0 + px) * scale, (y - cy0 + py) * scale)


def _clip_poly(poly: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip polygon points to image boundaries."""
    out = poly.copy()
    out[:, 0] = np.clip(out[:, 0], 0, max(0, width - 1))
    out[:, 1] = np.clip(out[:, 1], 0, max(0, height - 1))
    return out


def _resolve_global_path(preproc_root: Path, image_key: str) -> Path:
    """Resolve global image path, supporting both `global` and `global_1024` layouts."""
    p = preproc_root / "global" / f"{image_key}.png"
    if p.exists():
        return p
    p_1024 = preproc_root / "global_1024" / f"{image_key}.png"
    if p_1024.exists():
        return p_1024
    raise FileNotFoundError(f"Missing preprocessed global image for {image_key}")


def _parse_uwf_obb(
    label_path: Path,
    width: int,
    height: int,
    class_map: dict[str, str],
    allowed_classes: set[str],
    image_path: Path | None = None,
) -> list[dict]:
    """Parse UWF OBB txt labels as polygons in raw image pixel space."""
    anns: list[dict] = []
    if not label_path.exists():
        return anns
    if (width <= 0 or height <= 0) and image_path is not None and image_path.exists():
        raw = read_image(image_path)
        height, width = raw.shape[:2]

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        cls_name = class_map.get(str(int(float(parts[0]))))
        if not cls_name or cls_name not in allowed_classes:
            continue
        pts_norm = [float(v) for v in parts[1:]]
        poly = np.array(
            [[pts_norm[i] * width, pts_norm[i + 1] * height] for i in range(0, 8, 2)], dtype=np.float32
        )
        anns.append({"class_name": cls_name, "poly_raw": poly, "source": "uwf_obb"})
    return anns


def _parse_fgadr_masks(
    image_name: str,
    root: Path,
    class_map: dict[str, str],
    allowed_classes: set[str],
    min_area: int,
    simplify_eps: float,
) -> list[dict]:
    """Parse FGADR mask components as polygons in raw image pixel space."""
    anns: list[dict] = []
    for mask_dir in sorted(root.glob("*_Masks")):
        cls_name = class_map.get(mask_dir.name, mask_dir.name.replace("_Masks", "").lower())
        if cls_name not in allowed_classes:
            continue
        mask_path = mask_dir / image_name
        if not mask_path.exists():
            continue
        raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        mask = (raw > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) < int(min_area):
                continue
            comp = (labels == i).astype(np.uint8)
            cts = _contours(comp)
            if not cts:
                continue
            cnt = max(cts, key=cv2.contourArea)
            if float(cv2.contourArea(cnt)) < float(min_area):
                continue
            cnt = cnt.reshape(-1, 2).astype(np.float32)
            if simplify_eps > 0:
                cnt = cv2.approxPolyDP(cnt, epsilon=float(simplify_eps), closed=True).reshape(-1, 2).astype(np.float32)
            if cnt.shape[0] < 3:
                continue
            anns.append({"class_name": cls_name, "poly_raw": cnt, "source": "fgadr_mask"})
    return anns


def _serialize_object(poly_px: np.ndarray, width: int, height: int, class_id: int, class_name: str, source: str) -> dict | None:
    """Serialize polygon to normalized polygon/bbox/obb fields."""
    bbox_xyxy = _poly_bbox_xyxy(poly_px)
    if bbox_xyxy is None:
        return None
    obb_px = _poly_to_obb(poly_px)
    if obb_px is None:
        return None
    obb_poly = np.array(list(zip(obb_px[0::2], obb_px[1::2])), dtype=np.float32)
    area = float(abs(cv2.contourArea(poly_px.astype(np.float32))))
    if area <= 1.0:
        return None
    return {
        "class_id": int(class_id),
        "class_name": class_name,
        "source": source,
        "area_px": area,
        "polygon": _to_norm_poly(poly_px, width, height),
        "bbox_xywh": _to_norm_box_xywh(bbox_xyxy, width, height),
        "obb": _to_norm_poly(obb_poly, width, height),
    }


def _build_global_objects(
    row: dict,
    class_map_cfg: dict,
    allowed_classes: set[str],
    crop_meta: dict,
    global_meta: dict,
    global_size: tuple[int, int],
    min_comp_area: int,
    simplify_eps: float,
) -> list[dict]:
    """Build projected global objects for one manifest row."""
    categories = class_map_cfg["categories"]
    cat_to_id = {c: i + 1 for i, c in enumerate(categories)}
    uwf_map = {str(k): v for k, v in class_map_cfg["maps"].get("uwf700", {}).items()}
    fgadr_map = {str(k): v for k, v in class_map_cfg["maps"].get("fgadr", {}).items()}

    src_anns: list[dict] = []
    if row["dataset"] == "uwf700" and row.get("labels_path"):
        src_anns = _parse_uwf_obb(
            Path(row["labels_path"]),
            int(row.get("width", 0) or 0),
            int(row.get("height", 0) or 0),
            uwf_map,
            allowed_classes,
            Path(row["filepath"]),
        )
    elif row["dataset"] == "fgadr":
        src_anns = _parse_fgadr_masks(
            image_name=Path(row["filepath"]).name,
            root=Path(row["labels_path"]),
            class_map=fgadr_map,
            allowed_classes=allowed_classes,
            min_area=min_comp_area,
            simplify_eps=simplify_eps,
        )

    g_w, g_h = global_size
    out: list[dict] = []
    for ann in src_anns:
        pts = []
        for x, y in ann["poly_raw"].tolist():
            gx, gy = _project_point_to_global(float(x), float(y), crop_meta, global_meta)
            pts.append([gx, gy])
        poly = _clip_poly(np.array(pts, dtype=np.float32), g_w, g_h)
        if poly.shape[0] < 3:
            continue
        packed = _serialize_object(
            poly_px=poly,
            width=g_w,
            height=g_h,
            class_id=cat_to_id[ann["class_name"]],
            class_name=ann["class_name"],
            source=ann["source"],
        )
        if packed is not None:
            out.append(packed)
    return out


def _make_id_map(global_objects: list[dict], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize objects to id map and return per-object area pixels."""
    id_map = np.zeros((height, width), dtype=np.int32)
    for idx, obj in enumerate(global_objects, start=1):
        poly = np.array(list(zip(obj["polygon"][0::2], obj["polygon"][1::2])), dtype=np.float32)
        poly[:, 0] *= float(width)
        poly[:, 1] *= float(height)
        poly = np.round(poly).astype(np.int32)
        cv2.fillPoly(id_map, [poly], int(idx))
    flat = id_map.reshape(-1)
    areas = np.bincount(flat, minlength=len(global_objects) + 1)
    return id_map, areas


def _tile_objects(
    id_map: np.ndarray,
    global_objects: list[dict],
    tile_meta: dict,
    global_areas: np.ndarray,
    min_tile_obj_ratio: float,
    simplify_eps: float,
) -> list[dict]:
    """Extract clipped tile polygons from the global id map."""
    x0, y0, x1, y1 = [int(tile_meta[k]) for k in ("x0", "y0", "x1", "y1")]
    tile_w = max(1, x1 - x0)
    tile_h = max(1, y1 - y0)
    sub = id_map[y0:y1, x0:x1]

    out: list[dict] = []
    for oid in [int(v) for v in np.unique(sub) if int(v) > 0]:
        full_area = max(1.0, float(global_areas[oid]))
        inter = float((sub == oid).sum())
        if (inter / full_area) < float(min_tile_obj_ratio):
            continue
        comp = (sub == oid).astype(np.uint8)
        cts = _contours(comp)
        if not cts:
            continue
        for cnt in cts:
            if float(cv2.contourArea(cnt)) < 2.0:
                continue
            poly = cnt.reshape(-1, 2).astype(np.float32)
            if simplify_eps > 0:
                poly = cv2.approxPolyDP(poly, epsilon=float(simplify_eps), closed=True).reshape(-1, 2).astype(np.float32)
            if poly.shape[0] < 3:
                continue
            src = global_objects[oid - 1]
            packed = _serialize_object(
                poly_px=poly,
                width=tile_w,
                height=tile_h,
                class_id=int(src["class_id"]),
                class_name=str(src["class_name"]),
                source=str(src["source"]),
            )
            if packed is not None:
                out.append(packed)
    return out


def filter_class_map(class_map_cfg: dict, allowed_classes: set[str]) -> dict:
    """Return a class map config filtered to allowed class names."""
    categories = [c for c in class_map_cfg["categories"] if c in allowed_classes]
    out = {"categories": categories, "maps": {}}
    for ds_name, mapping in class_map_cfg.get("maps", {}).items():
        out_map = {}
        for key, cls in mapping.items():
            if cls in allowed_classes:
                out_map[key] = cls
        out["maps"][ds_name] = out_map
    return out


def build_native_labels_from_manifest(
    manifest_rows: list[dict],
    split_ids: set[str],
    class_map_cfg: dict,
    preproc_root: Path,
    out_root: Path,
    dataset_name: str,
    split_name: str,
    include_global: bool,
    include_tiles: bool,
    min_comp_area: int,
    min_tile_obj_ratio: float,
    simplify_eps: float,
) -> dict:
    """Build native (non-COCO) polygon labels for one dataset/split."""
    categories = class_map_cfg["categories"]
    allowed = set(categories)

    records_root = ensure_dir(out_root / "records" / dataset_name / split_name)
    global_rec_dir = ensure_dir(records_root / "global")
    tile_rec_dir = ensure_dir(records_root / "tiles")

    global_rows: list[dict] = []
    tile_rows: list[dict] = []
    num_global_objects = 0
    num_tile_objects = 0

    for row in manifest_rows:
        if row["image_id"] not in split_ids:
            continue
        image_key = row["image_id"].replace("::", "__")
        crop_meta_path = preproc_root / "crop_meta" / f"{image_key}.json"
        tiles_meta_path = preproc_root / "tiles_meta" / f"{image_key}.json"
        if not crop_meta_path.exists() or not tiles_meta_path.exists():
            continue
        global_img = _resolve_global_path(preproc_root, image_key)
        crop_meta = json.loads(crop_meta_path.read_text(encoding="utf-8"))
        tiles_meta = json.loads(tiles_meta_path.read_text(encoding="utf-8"))

        g_w, g_h = [int(v) for v in tiles_meta["global_size"]]
        global_objects = _build_global_objects(
            row=row,
            class_map_cfg=class_map_cfg,
            allowed_classes=allowed,
            crop_meta=crop_meta,
            global_meta=tiles_meta["global_meta"],
            global_size=(g_w, g_h),
            min_comp_area=min_comp_area,
            simplify_eps=simplify_eps,
        )

        if include_global:
            rec = {
                "image_id": row["image_id"],
                "dataset": row["dataset"],
                "split": split_name,
                "image_path": global_img.as_posix(),
                "width": g_w,
                "height": g_h,
                "objects": global_objects,
            }
            rec_path = global_rec_dir / f"{image_key}.json"
            save_json(rec_path, rec)
            global_rows.append(
                {
                    "record_id": f"{row['image_id']}::global",
                    "image_id": row["image_id"],
                    "dataset": row["dataset"],
                    "split": split_name,
                    "scope": "global",
                    "image_path": global_img.as_posix(),
                    "labels_path": rec_path.as_posix(),
                    "width": g_w,
                    "height": g_h,
                    "num_objects": len(global_objects),
                }
            )
            num_global_objects += len(global_objects)

        if not include_tiles:
            continue

        id_map, areas = _make_id_map(global_objects, g_w, g_h)
        tile_img_root = preproc_root / "tiles" / image_key
        tile_row_dir = ensure_dir(tile_rec_dir / image_key)
        for tile_meta in tiles_meta["tiles"]:
            tile_id = str(tile_meta["tile_id"])
            tile_path = tile_img_root / f"{tile_id}.png"
            if not tile_path.exists():
                continue
            t_w = int(tile_meta["x1"] - tile_meta["x0"])
            t_h = int(tile_meta["y1"] - tile_meta["y0"])
            tile_objects = _tile_objects(
                id_map=id_map,
                global_objects=global_objects,
                tile_meta=tile_meta,
                global_areas=areas,
                min_tile_obj_ratio=min_tile_obj_ratio,
                simplify_eps=simplify_eps,
            )
            rec = {
                "image_id": row["image_id"],
                "dataset": row["dataset"],
                "split": split_name,
                "scope": "tile",
                "tile_id": tile_id,
                "tile_box_xyxy_global": [
                    int(tile_meta["x0"]),
                    int(tile_meta["y0"]),
                    int(tile_meta["x1"]),
                    int(tile_meta["y1"]),
                ],
                "image_path": tile_path.as_posix(),
                "width": t_w,
                "height": t_h,
                "objects": tile_objects,
            }
            rec_path = tile_row_dir / f"{tile_id}.json"
            save_json(rec_path, rec)
            tile_rows.append(
                {
                    "record_id": f"{row['image_id']}::{tile_id}",
                    "image_id": row["image_id"],
                    "dataset": row["dataset"],
                    "split": split_name,
                    "scope": "tile",
                    "tile_id": tile_id,
                    "image_path": tile_path.as_posix(),
                    "labels_path": rec_path.as_posix(),
                    "width": t_w,
                    "height": t_h,
                    "num_objects": len(tile_objects),
                }
            )
            num_tile_objects += len(tile_objects)

    if include_global:
        save_jsonl(out_root / f"{dataset_name}_{split_name}_global.jsonl", global_rows)
    if include_tiles:
        save_jsonl(out_root / f"{dataset_name}_{split_name}_tiles.jsonl", tile_rows)

    return {
        "dataset": dataset_name,
        "split": split_name,
        "num_global_records": len(global_rows),
        "num_tile_records": len(tile_rows),
        "num_global_objects": num_global_objects,
        "num_tile_objects": num_tile_objects,
    }
