from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .common import ensure_dir, read_image, save_json, write_image


def compute_roi_mask(image: np.ndarray, cfg: dict) -> np.ndarray:
    max_side = int(cfg.get("downsample_max_side", 768))
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    _, t_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sat_thr = int(cfg.get("sat_threshold", 18))
    mask = ((sat > sat_thr) | (t_gray > 0)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros((h, w), dtype=np.uint8)

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def crop_to_roi(image: np.ndarray, mask: np.ndarray, pad_px: int = 12) -> tuple[np.ndarray, dict]:
    ys, xs = np.where(mask > 0)
    h, w = image.shape[:2]
    if len(xs) == 0 or len(ys) == 0:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0 = max(0, int(xs.min()) - pad_px)
        y0 = max(0, int(ys.min()) - pad_px)
        x1 = min(w, int(xs.max()) + pad_px + 1)
        y1 = min(h, int(ys.max()) + pad_px + 1)
    crop = image[y0:y1, x0:x1]
    meta = {
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "orig_size": [int(w), int(h)],
        "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
    }
    return crop, meta


def normalize_color(image: np.ndarray, roi_mask: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    mask = roi_mask > 0
    out = image.astype(np.float32).copy()
    roi_px = out[mask]
    if roi_px.size == 0:
        return image.copy(), {"method": method, "mean": [0, 0, 0], "std": [0, 0, 0]}

    mean = roi_px.mean(axis=0)
    std = roi_px.std(axis=0) + 1e-6

    if method == "zscore_rgb":
        norm = (out - mean[None, None, :]) / std[None, None, :]
        norm = np.clip(norm, -2.5, 2.5)
        norm = ((norm + 2.5) / 5.0) * 255.0
        out = norm
    elif method == "grayworld":
        target = float(mean.mean())
        gains = target / std.clip(min=1e-6)
        out = out * gains[None, None, :]
    elif method == "clahe_luminance":
        lab = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(lab[:, :, 0])
        lab[:, :, 0] = l
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    out[~mask] = 0.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    meta = {
        "method": method,
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
    }
    return out, meta


def resize_global(image: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    side = max(h, w)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + h, pad_x : pad_x + w] = image
    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    meta = {
        "pad_xy": [int(pad_x), int(pad_y)],
        "padded_side": int(side),
        "scale": float(size / side),
        "orig_size": [int(w), int(h)],
    }
    return resized, meta


def tile(image: np.ndarray, tile_size: int, overlap: float) -> tuple[list[np.ndarray], list[dict]]:
    h, w = image.shape[:2]
    stride = max(1, int(tile_size * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    if not xs or xs[-1] != w - tile_size:
        xs.append(max(0, w - tile_size))
    if not ys or ys[-1] != h - tile_size:
        ys.append(max(0, h - tile_size))

    tiles: list[np.ndarray] = []
    metas: list[dict] = []
    idx = 0
    for y0 in ys:
        for x0 in xs:
            x1 = min(w, x0 + tile_size)
            y1 = min(h, y0 + tile_size)
            tile_img = image[y0:y1, x0:x1]
            if tile_img.shape[0] != tile_size or tile_img.shape[1] != tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[: tile_img.shape[0], : tile_img.shape[1]] = tile_img
                tile_img = padded
            tiles.append(tile_img)
            metas.append(
                {
                    "tile_id": f"tile_{idx:03d}",
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(min(w, x0 + tile_size)),
                    "y1": int(min(h, y0 + tile_size)),
                    "scale": 1.0,
                }
            )
            idx += 1
    return tiles, metas


def reconstruct_from_tiles(tiles: list[np.ndarray], metas: list[dict], out_size: tuple[int, int]) -> np.ndarray:
    h, w = out_size
    acc = np.zeros((h, w, 3), dtype=np.float32)
    cnt = np.zeros((h, w, 1), dtype=np.float32)
    for tile_img, meta in zip(tiles, metas):
        x0, y0, x1, y1 = meta["x0"], meta["y0"], meta["x1"], meta["y1"]
        view = tile_img[: y1 - y0, : x1 - x0].astype(np.float32)
        acc[y0:y1, x0:x1] += view
        cnt[y0:y1, x0:x1] += 1
    cnt[cnt == 0] = 1
    return (acc / cnt).astype(np.uint8)


def process_manifest(manifest_rows: list[dict], cfg: dict, out_root: Path) -> dict:
    roi_dir = ensure_dir(out_root / "roi_masks")
    crop_dir = ensure_dir(out_root / "crops")
    crop_meta_dir = ensure_dir(out_root / "crop_meta")
    norm_dir = ensure_dir(out_root / "norm")
    norm_meta_dir = ensure_dir(out_root / "norm_meta")
    global_dir = ensure_dir(out_root / "global_1024")
    tiles_dir = ensure_dir(out_root / "tiles")
    tiles_meta_dir = ensure_dir(out_root / "tiles_meta")

    fail_small = 0
    fail_border = 0

    for row in manifest_rows:
        image_id = row["image_id"].replace("::", "__")
        image = read_image(row["filepath"])

        mask = compute_roi_mask(image, cfg["roi"])
        write_image(roi_dir / f"{image_id}.png", np.repeat(mask[:, :, None], 3, axis=2))

        crop, crop_meta = crop_to_roi(image, mask, int(cfg["roi"].get("crop_pad_px", 12)))
        write_image(crop_dir / f"{image_id}.png", crop)
        save_json(crop_meta_dir / f"{image_id}.json", crop_meta)

        roi_crop = mask[
            crop_meta["bbox_xyxy"][1] : crop_meta["bbox_xyxy"][3],
            crop_meta["bbox_xyxy"][0] : crop_meta["bbox_xyxy"][2],
        ]
        norm, norm_meta = normalize_color(crop, roi_crop, cfg["normalize"]["method"])
        write_image(norm_dir / f"{image_id}.png", norm)
        save_json(norm_meta_dir / f"{image_id}.json", norm_meta)

        global_img, global_meta = resize_global(norm, int(cfg["resize"]["global_size"]))
        write_image(global_dir / f"{image_id}.png", global_img)

        tiles, metas = tile(global_img, int(cfg["tiling"]["tile_size"]), float(cfg["tiling"]["overlap"]))
        tile_img_dir = ensure_dir(tiles_dir / image_id)
        for tile_img, meta in zip(tiles, metas):
            write_image(tile_img_dir / f"{meta['tile_id']}.png", tile_img)

        save_json(
            tiles_meta_dir / f"{image_id}.json",
            {
                "image_id": row["image_id"],
                "global_size": [int(global_img.shape[1]), int(global_img.shape[0])],
                "global_meta": global_meta,
                "tiles": metas,
            },
        )

        roi_area_ratio = float((mask > 0).mean())
        x0, y0, x1, y1 = crop_meta["bbox_xyxy"]
        border_touch = x0 == 0 or y0 == 0 or x1 == image.shape[1] or y1 == image.shape[0]
        if roi_area_ratio < float(cfg["verify"]["min_roi_area_ratio"]):
            fail_small += 1
        if border_touch:
            fail_border += 1

    n = max(1, len(manifest_rows))
    return {
        "num_images": len(manifest_rows),
        "roi_small_rate": fail_small / n,
        "roi_border_touch_rate": fail_border / n,
    }
