#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import random
import shutil

import cv2
import numpy as np

from uveitis_pipeline.common import load_yaml, read_image, write_image, ensure_dir
from uveitis_pipeline.preprocess import (
    Sam2PromptMasker,
    compute_roi_mask,
    crop_to_roi,
    normalize_color,
    resize_global,
    tile,
)


def _overlay_boundary(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask > 0
    edges = cv2.Canny((m.astype(np.uint8) * 255), 80, 160)
    out = image.copy()
    out[edges > 0] = [255, 0, 0]
    return out


def _preview(image: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _pad_to_height(image: np.ndarray, h: int) -> np.ndarray:
    if image.shape[0] == h:
        return image
    if image.shape[0] > h:
        return image[:h]
    pad = np.zeros((h - image.shape[0], image.shape[1], 3), dtype=image.dtype)
    return np.concatenate([image, pad], axis=0)


def _safe_erode(mask: np.ndarray, erode_px: int) -> np.ndarray:
    if erode_px <= 0:
        return mask
    k = int(erode_px) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded if np.any(eroded > 0) else mask


def _pick_diverse(images_root: Path, n: int, seed: int = 42) -> list[Path]:
    rng = random.Random(seed)
    buckets: dict[str, list[Path]] = {}
    for d in sorted(images_root.iterdir()):
        if not d.is_dir():
            continue
        imgs = [p for p in sorted(d.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
        if imgs:
            buckets[d.name] = imgs

    keys = list(buckets.keys())
    if not keys:
        return []

    picked: list[Path] = []
    i = 0
    while len(picked) < n:
        k = keys[i % len(keys)]
        pool = buckets[k]
        if pool:
            idx = rng.randrange(len(pool))
            picked.append(pool.pop(idx))
        i += 1
        if i > 10_000:
            break
    return picked[:n]


def main() -> None:
    p = argparse.ArgumentParser(description="QA: SAM2 fundus masking + crop + ref color normalization on diverse UWF images.")
    p.add_argument("--config", default="configs/stage0_preprocess.yaml")
    p.add_argument("--images-root", default="datasets/uwf-700/Images")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default="eval/preproc_norm_qa_uwf20")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    cfg = load_yaml(args.config)
    roi_cfg = cfg["roi"]
    norm_cfg = cfg["normalize"]
    norm_method = str(norm_cfg.get("method", "zscore_rgb"))
    erode_px = int(norm_cfg.get("stats_erode_px", 4))

    ref = None
    if norm_method == "reinhard_lab_ref":
        ref_path = Path(norm_cfg.get("ref", {}).get("stats_path", ""))
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Missing color reference stats: {ref_path}. Run scripts/build_regular_fundus_color_ref.py first."
            )
        ref = json.loads(ref_path.read_text(encoding="utf-8"))

    sam2_masker = Sam2PromptMasker(roi_cfg.get("sam2", {}))

    images_root = Path(args.images_root)
    picked = _pick_diverse(images_root, int(args.n), seed=42)
    if not picked:
        raise RuntimeError(f"No images found under {images_root}")

    dirs = {
        "overlays": ensure_dir(out_dir / "overlays"),
        "masks": ensure_dir(out_dir / "masks"),
        "masked_raw": ensure_dir(out_dir / "masked_raw"),
        "crop": ensure_dir(out_dir / "crop"),
        "norm": ensure_dir(out_dir / "norm"),
        "global_1024": ensure_dir(out_dir / "global_1024"),
        "tiles": ensure_dir(out_dir / "tiles"),
        "triptychs": ensure_dir(out_dir / "triptychs"),
    }

    rows = []
    for img_path in picked:
        image = read_image(img_path)
        mask = compute_roi_mask(image, roi_cfg, dataset="uwf700", sam2_masker=sam2_masker)
        mask_u8 = (mask > 0).astype(np.uint8) * 255

        overlay = _overlay_boundary(image, mask_u8)
        masked_raw = image.copy()
        masked_raw[mask_u8 == 0] = 0

        crop, crop_meta = crop_to_roi(image, mask_u8, int(roi_cfg.get("crop_pad_px", 12)))
        x0, y0, x1, y1 = crop_meta["bbox_xyxy"]
        roi_crop = mask_u8[y0:y1, x0:x1]
        stats_mask = _safe_erode(roi_crop, erode_px)
        norm, norm_meta = normalize_color(crop, stats_mask, norm_method, out_mask=roi_crop, ref=ref)

        global_img, global_meta = resize_global(norm, int(cfg["resize"]["global_size"]))
        tiles, metas = tile(global_img, int(cfg["tiling"]["tile_size"]), float(cfg["tiling"]["overlap"]))

        key = f"{img_path.parent.name}__{img_path.stem}"
        write_image(dirs["overlays"] / f"{key}.png", overlay)
        write_image(dirs["masks"] / f"{key}.png", np.repeat(mask_u8[:, :, None], 3, axis=2))
        write_image(dirs["masked_raw"] / f"{key}.png", masked_raw)
        write_image(dirs["crop"] / f"{key}.png", crop)
        write_image(dirs["norm"] / f"{key}.png", norm)
        write_image(dirs["global_1024"] / f"{key}.png", global_img)
        a, b, c = _preview(overlay), _preview(masked_raw), _preview(norm)
        hh = max(a.shape[0], b.shape[0], c.shape[0])
        tri = np.concatenate([_pad_to_height(a, hh), _pad_to_height(b, hh), _pad_to_height(c, hh)], axis=1)
        write_image(dirs["triptychs"] / f"{key}.png", tri)

        tile_dir = ensure_dir(dirs["tiles"] / key)
        for tile_img, meta in zip(tiles, metas):
            write_image(tile_dir / f"{meta['tile_id']}.png", tile_img)

        roi_area = float((mask_u8 > 0).mean())
        rows.append(
            {
                "key": key,
                "path": str(img_path),
                "roi_area_ratio": roi_area,
                "crop_meta": crop_meta,
                "norm_meta": norm_meta,
                "global_meta": global_meta,
                "n_tiles": len(metas),
            }
        )

    (out_dir / "selected_images.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
