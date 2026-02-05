#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uveitis_pipeline.common import ensure_dir, load_yaml, read_image, save_json, write_image
from uveitis_pipeline.preprocess import Sam2PromptMasker, SamPromptMasker, compute_roi_mask


def _sort_key(path: Path):
    m = re.search(r"(\d+)$", path.stem)
    return (path.parent.as_posix(), int(m.group(1)) if m else path.stem)


def _overlay_boundary(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(mask.astype(np.uint8), 80, 160)
    out = image.copy()
    out[edges > 0] = [255, 0, 0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM2 fundus mask QA (save overlays/masks for N UWF images)")
    ap.add_argument("--images", type=Path, default=Path("datasets/uwf-700/Images"))
    ap.add_argument("--config", type=Path, default=Path("configs/stage0_preprocess.yaml"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, default=Path("eval/sam2_fundus_qa"))
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    roi_cfg = cfg["roi"]

    use_sam = (
        roi_cfg.get("method") == "sam_prompted"
        or roi_cfg.get("method_by_dataset", {}).get("uwf700") == "sam_prompted"
    )
    use_sam2 = (
        roi_cfg.get("method") == "sam2_prompted"
        or roi_cfg.get("method_by_dataset", {}).get("uwf700") == "sam2_prompted"
    )

    sam_masker = None
    if use_sam:
        try:
            sam_masker = SamPromptMasker(roi_cfg.get("sam", {}))
        except Exception as e:
            print(f"SAM (v1) unavailable, using fallback: {e}")

    sam2_masker = None
    if use_sam2:
        try:
            sam2_masker = Sam2PromptMasker(roi_cfg.get("sam2", {}))
        except Exception as e:
            print(f"SAM2 unavailable, using fallback: {e}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted([p for p in args.images.rglob("*") if p.suffix.lower() in exts], key=_sort_key)
    if not paths:
        raise SystemExit(f"No images found under {args.images}")
    paths = paths[: max(0, int(args.n))]

    out_dir = ensure_dir(args.out_dir)
    masks_dir = ensure_dir(out_dir / "masks")
    overlays_dir = ensure_dir(out_dir / "overlays")
    masked_dir = ensure_dir(out_dir / "masked")

    rows = []
    for p in paths:
        img = read_image(p)
        mask = compute_roi_mask(img, roi_cfg, dataset="uwf700", sam_masker=sam_masker, sam2_masker=sam2_masker)
        area = float((mask > 0).mean())
        key = p.relative_to(args.images).as_posix().replace("/", "__")

        over = _overlay_boundary(img, mask)
        masked = img.copy()
        masked[mask == 0] = 0

        write_image(overlays_dir / f"{key}.png", over)
        write_image(masked_dir / f"{key}.png", masked)
        cv2.imwrite(str(masks_dir / f"{key}.png"), mask.astype(np.uint8))

        rows.append({"path": p.as_posix(), "key": key, "mask_area_ratio": area})

    save_json(out_dir / "metrics.json", {"n": len(rows), "mean_area_ratio": float(np.mean([r["mask_area_ratio"] for r in rows])), "rows": rows})
    print(f"Wrote {len(rows)} overlays to {overlays_dir}")
    print(f"Wrote {len(rows)} masks to {masks_dir}")


if __name__ == "__main__":
    main()

