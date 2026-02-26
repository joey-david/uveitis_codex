#!/usr/bin/env python3
"""Render GT (red) + prediction (green) overlays from native labels/predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uveitis_pipeline.common import ensure_dir, read_jsonl


def _poly_to_px(poly: list[float], w: int, h: int) -> np.ndarray:
    """Convert normalized polygon list to Nx2 int32 points."""
    pts = [[int(np.clip(poly[i] * w, 0, w - 1)), int(np.clip(poly[i + 1] * h, 0, h - 1))] for i in range(0, len(poly), 2)]
    return np.array(pts, dtype=np.int32)


def main() -> None:
    """Create overlay previews for a native index and prediction jsonl."""
    parser = argparse.ArgumentParser(description="QA overlays for native predictions")
    parser.add_argument("--gt-index", required=True)
    parser.add_argument("--pred-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-images", type=int, default=30)
    args = parser.parse_args()

    rows = read_jsonl(args.gt_index)
    pred_rows = {str(r.get("record_id", "")): r for r in read_jsonl(args.pred_jsonl)}
    out_dir = ensure_dir(Path(args.out_dir))

    for i, row in enumerate(rows[: max(0, int(args.max_images))] if args.max_images > 0 else rows):
        img = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        gt = json.loads(Path(row["labels_path"]).read_text(encoding="utf-8"))
        for obj in gt.get("objects", []):
            pts = _poly_to_px(obj["polygon"], w=w, h=h)
            if pts.shape[0] < 3:
                continue
            cv2.polylines(img, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
            cv2.putText(img, f"gt:{obj['class_name']}", (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        for pred in pred_rows.get(str(row.get("record_id", "")), {}).get("predictions", []):
            pts = _poly_to_px(pred.get("polygon", []), w=w, h=h)
            if pts.shape[0] < 3:
                continue
            cv2.polylines(img, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
            cv2.putText(
                img,
                f"pr:{pred['class_name']}:{float(pred.get('score', 0.0)):.2f}",
                (x, min(h - 1, y + 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        out_path = out_dir / f"{i:03d}__{Path(row['image_path']).stem}.png"
        cv2.imwrite(str(out_path), img)


if __name__ == "__main__":
    main()
