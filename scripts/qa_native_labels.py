#!/usr/bin/env python3
"""Visual QA for native polygon labels."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import random

import cv2
import numpy as np

from uveitis_pipeline.common import ensure_dir, read_jsonl, save_json, write_image


def _draw(image: np.ndarray, objects: list[dict]) -> np.ndarray:
    """Draw polygon overlays with class labels."""
    out = image.copy()
    h, w = out.shape[:2]
    for obj in objects:
        poly = np.array(list(zip(obj["polygon"][0::2], obj["polygon"][1::2])), dtype=np.float32)
        poly[:, 0] = np.clip(poly[:, 0] * w, 0, max(0, w - 1))
        poly[:, 1] = np.clip(poly[:, 1] * h, 0, max(0, h - 1))
        poly_i = np.round(poly).astype(np.int32)
        cv2.polylines(out, [poly_i], True, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        x, y = int(poly_i[:, 0].min()), int(poly_i[:, 1].min())
        cv2.putText(out, obj["class_name"], (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return out


def _mask(image: np.ndarray, objects: list[dict]) -> np.ndarray:
    """Build binary mask from all polygons for visualization."""
    h, w = image.shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    for obj in objects:
        poly = np.array(list(zip(obj["polygon"][0::2], obj["polygon"][1::2])), dtype=np.float32)
        poly[:, 0] = np.clip(poly[:, 0] * w, 0, max(0, w - 1))
        poly[:, 1] = np.clip(poly[:, 1] * h, 0, max(0, h - 1))
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
    return np.repeat(m[:, :, None], 3, axis=2)


def main() -> None:
    """Create overlay/mask QA images for a native label index."""
    p = argparse.ArgumentParser(description="QA native labels")
    p.add_argument("--index", required=True)
    p.add_argument("--out", default="eval/native_labels_qa")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = read_jsonl(args.index)
    rows = [r for r in rows if int(r.get("num_objects", 0)) > 0]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: max(0, int(args.n))]

    out_dir = ensure_dir(Path(args.out))
    overlays = ensure_dir(out_dir / "overlays")
    masks = ensure_dir(out_dir / "masks")

    report = []
    for row in rows:
        img = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rec = json.loads(Path(row["labels_path"]).read_text(encoding="utf-8"))
        objs = rec.get("objects", [])

        key = row["record_id"].replace("::", "__")
        over = _draw(img, objs)
        mask = _mask(img, objs)

        write_image(overlays / f"{key}.png", over)
        write_image(masks / f"{key}.png", mask)

        report.append(
            {
                "record_id": row["record_id"],
                "image_id": row.get("image_id", ""),
                "num_objects": len(objs),
                "classes": sorted({o["class_name"] for o in objs}),
            }
        )

    save_json(out_dir / "qa_summary.json", {"n": len(report), "rows": report})
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
