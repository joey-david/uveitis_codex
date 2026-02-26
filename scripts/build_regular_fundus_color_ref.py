#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from datetime import datetime, timezone
import random

import cv2
import numpy as np

from uveitis_pipeline.common import load_yaml, read_image, read_jsonl, save_json
from uveitis_pipeline.preprocess import compute_roi_mask


def _safe_erode(mask: np.ndarray, erode_px: int) -> np.ndarray:
    if erode_px <= 0:
        return mask
    k = int(erode_px) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded if np.any(eroded > 0) else mask


def main() -> None:
    p = argparse.ArgumentParser(description="Compute regular-fundus color reference stats (masked, non-black pixels).")
    p.add_argument("--config", default="configs/stage0_preprocess.yaml")
    p.add_argument("--out", default="preproc/ref/regular_fundus_color_stats.json")
    p.add_argument("--per-dataset", type=int, default=50)
    p.add_argument("--max-images", type=int, default=0, help="Optional cap after per-dataset sampling")
    p.add_argument("--downsample-max-side", type=int, default=512)
    p.add_argument("--pixels-per-image", type=int, default=20000)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    manifests = cfg["input"]["manifests"]
    roi_cfg = cfg["roi"]
    norm_cfg = cfg.get("normalize", {})
    erode_px = int(norm_cfg.get("stats_erode_px", 4))

    rows = []
    for m in manifests:
        rows.extend(read_jsonl(m))

    regular_datasets = {"eyepacs", "fgadr", "deepdrid_regular"}
    rows = [r for r in rows if r.get("dataset") in regular_datasets]
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(str(r.get("dataset")), []).append(r)

    sampled: list[dict] = []
    rnd = random.Random(42)
    per_ds = int(args.per_dataset)
    for ds, ds_rows in sorted(by_ds.items()):
        if per_ds > 0 and len(ds_rows) > per_ds:
            sampled.extend(rnd.sample(ds_rows, k=per_ds))
        else:
            sampled.extend(ds_rows)
    rows = sampled
    if args.max_images > 0:
        rows = rows[: args.max_images]

    lab_sum = np.zeros(3, dtype=np.float64)
    lab_sumsq = np.zeros(3, dtype=np.float64)
    rgb_sum = np.zeros(3, dtype=np.float64)
    rgb_sumsq = np.zeros(3, dtype=np.float64)
    n_px = 0
    n_img = 0
    rng = np.random.default_rng(42)

    for r in rows:
        img = read_image(r["filepath"])
        max_side = int(args.downsample_max_side)
        scale = 1.0
        if max_side > 0:
            h, w = img.shape[:2]
            scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        mask = compute_roi_mask(img, roi_cfg, dataset=r.get("dataset", ""))
        stats = _safe_erode((mask > 0).astype(np.uint8) * 255, erode_px) > 0
        stats &= (img.astype(np.int32).sum(axis=2) > 0)
        if not np.any(stats):
            continue

        ys, xs = np.where(stats)
        if args.pixels_per_image > 0 and len(xs) > int(args.pixels_per_image):
            idx = rng.choice(len(xs), size=int(args.pixels_per_image), replace=False)
            ys, xs = ys[idx], xs[idx]
        px_rgb = img[ys, xs].astype(np.float64)
        rgb_sum += px_rgb.sum(axis=0)
        rgb_sumsq += (px_rgb * px_rgb).sum(axis=0)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        px_lab = lab[ys, xs].astype(np.float64)
        lab_sum += px_lab.sum(axis=0)
        lab_sumsq += (px_lab * px_lab).sum(axis=0)

        n_px += int(px_rgb.shape[0])
        n_img += 1
        if n_img % 1000 == 0:
            print(json.dumps({"images": n_img, "pixels": n_px}))

    if n_px == 0:
        raise RuntimeError("No ROI pixels found while computing reference stats. Check dataset roots/manifests.")

    rgb_mean = rgb_sum / n_px
    rgb_var = np.maximum(1e-6, (rgb_sumsq / n_px) - (rgb_mean * rgb_mean))
    rgb_std = np.sqrt(rgb_var)

    lab_mean = lab_sum / n_px
    lab_var = np.maximum(1e-6, (lab_sumsq / n_px) - (lab_mean * lab_mean))
    lab_std = np.sqrt(lab_var)

    out = {
        "kind": "regular_fundus_color_reference",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": sorted(list(regular_datasets)),
        "n_images_used": int(n_img),
        "n_pixels": int(n_px),
        "rgb_mean": [float(x) for x in rgb_mean],
        "rgb_std": [float(x) for x in rgb_std],
        "lab_mean": [float(x) for x in lab_mean],
        "lab_std": [float(x) for x in lab_std],
        "cfg": {"stats_erode_px": int(erode_px), "roi_method": str(roi_cfg.get("method", "threshold"))},
    }

    save_json(args.out, out)
    print(json.dumps({"out": args.out, "n_images_used": n_img, "n_pixels": n_px}, indent=2))


if __name__ == "__main__":
    main()
