#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser(description="Render COCO bbox overlays for quick QA")
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = json.loads(args.coco.read_text(encoding="utf-8"))
    imgs = data["images"]
    cats = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}

    by_img = {}
    for a in data["annotations"]:
        by_img.setdefault(int(a["image_id"]), []).append(a)

    pool = [im for im in imgs if int(im["id"]) in by_img]
    random.seed(args.seed)
    sample = random.sample(pool, k=min(len(pool), int(args.n)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for im in sample:
        path = Path(im["file_name"])
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for a in by_img.get(int(im["id"]), []):
            x, y, w, h = a["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            name = cats.get(int(a["category_id"]), str(a["category_id"]))
            cv2.putText(img, name, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        prefix = path.parent.name if path.parent.name else "img"
        out = args.out_dir / f"{prefix}__{path.stem}_overlay.png"
        cv2.imwrite(str(out), img)
    print(f"Wrote overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
