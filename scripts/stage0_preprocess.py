#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np

from uveitis_pipeline.common import load_yaml, read_jsonl, save_json, read_image
from uveitis_pipeline.preprocess import process_manifest, reconstruct_from_tiles

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _overlay_boundary(image: np.ndarray, mask_rgb: np.ndarray) -> np.ndarray:
    mask = mask_rgb[:, :, 0] > 0
    edges = cv2.Canny((mask.astype(np.uint8) * 255), 80, 160)
    out = image.copy()
    out[edges > 0] = [255, 0, 0]
    return out


def _preview(image: np.ndarray, max_side: int = 640) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _global_path(preproc_root: Path, image_id: str) -> Path:
    """Resolve global image path from current/legacy preprocess layouts."""
    p = preproc_root / "global" / f"{image_id}.png"
    if p.exists():
        return p
    return preproc_root / "global_1024" / f"{image_id}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 0 preprocessing")
    parser.add_argument("--config", default="configs/stage0_preprocess.yaml")
    parser.add_argument("--manifest", default=None, help="Optional single manifest override")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    manifests = [args.manifest] if args.manifest else cfg["input"]["manifests"]

    rows = []
    for m in manifests:
        rows.extend(read_jsonl(m))

    if args.max_images > 0:
        rows = rows[: args.max_images]

    out_root = Path(cfg["output"]["preproc_root"])
    stats = process_manifest(rows, cfg, out_root)

    # Verification 1: montage overlays
    random.seed(42)
    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r)

    verify_dir = out_root / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    for dataset, ds_rows in by_dataset.items():
        sample = random.sample(ds_rows, k=min(int(cfg["verify"]["montage_n"]), len(ds_rows)))
        cols = 5
        rows_n = max(1, int(np.ceil(len(sample) / cols)))
        fig, axes = plt.subplots(rows_n, cols, figsize=(16, 3 * rows_n))
        axes = np.array(axes).reshape(-1)
        for i, row in enumerate(sample):
            key = row["image_id"].replace("::", "__")
            raw = read_image(row["filepath"])
            mask = read_image(out_root / "roi_masks" / f"{key}.png")
            view = _preview(_overlay_boundary(raw, mask))
            axes[i].imshow(view)
            axes[i].set_title(key, fontsize=8)
            axes[i].axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(verify_dir / f"{dataset}_roi_overlay_montage.png", dpi=170)
        plt.close(fig)

    # Verification 2: tile reconstruction sanity + tile coverage report
    tile_counts = []
    recon_error = []
    reconstruct_n = int(cfg["verify"].get("reconstruct_n", 50))
    for meta_file in sorted((out_root / "tiles_meta").glob("*.json"))[:reconstruct_n]:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        tile_counts.append(len(meta["tiles"]))
        g_w, g_h = meta["global_size"]
        image_id = meta["image_id"].replace("::", "__")
        global_img = read_image(_global_path(out_root, image_id))
        tiles = [read_image(out_root / "tiles" / image_id / f"{t['tile_id']}.png") for t in meta["tiles"]]
        recon = reconstruct_from_tiles(tiles, meta["tiles"], (g_h, g_w))
        recon_error.append(float(np.mean(np.abs(global_img.astype(np.float32) - recon.astype(np.float32)))))

    report = {
        **stats,
        "tile_count_distribution": dict(Counter(tile_counts)),
        "avg_reconstruction_abs_error": float(np.mean(recon_error)) if recon_error else 0.0,
    }
    save_json(out_root / "verify" / "preprocess_metrics.json", report)

    print("Preprocessing complete")
    print(report)


if __name__ == "__main__":
    main()
