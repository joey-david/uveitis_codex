#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fit_h(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / max(1, h)
    return cv2.resize(img, (int(round(w * scale)), target_h), interpolation=cv2.INTER_AREA)


def _parse_uwf_obb(label_path: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        cls = int(float(parts[0]))
        pts = [float(v) for v in parts[1:]]
        poly = list(zip(pts[0::2], pts[1::2]))
        out.append((cls, poly))
    return out


def _draw_raw_polys(raw_bgr: np.ndarray, polys_norm: list[tuple[int, list[tuple[float, float]]]]) -> np.ndarray:
    h, w = raw_bgr.shape[:2]
    out = raw_bgr.copy()
    for cls, poly in polys_norm:
        pts = np.array([[int(round(x * w)), int(round(y * h))] for x, y in poly], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 200, 255), thickness=3)
        x0, y0 = pts[:, 0, 0].min(), pts[:, 0, 1].min()
        cv2.putText(out, str(cls), (int(x0), int(max(0, y0 - 8))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    return out


def _draw_coco_boxes(
    img_bgr: np.ndarray,
    anns: list[dict],
    cat_id_to_name: dict[int, str],
) -> np.ndarray:
    out = img_bgr.copy()
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        name = cat_id_to_name.get(int(ann["category_id"]), str(ann["category_id"]))
        cv2.putText(out, name, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return out


def _nonblack_ratio(img_bgr: np.ndarray, box_xywh: list[float]) -> float:
    x, y, w, h = [int(v) for v in box_xywh]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img_bgr.shape[1], x + w), min(img_bgr.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = img_bgr[y0:y1, x0:x1]
    nonblack = np.any(crop > 8, axis=2)
    return float(nonblack.mean()) if nonblack.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="QA: show raw OBB labels vs projected COCO boxes on preprocessed images")
    ap.add_argument("--coco", type=Path, default=Path("labels_coco/uwf700_train.json"), help="COCO json path")
    ap.add_argument("--split-name", default="train", help="Used for output folder naming only")
    ap.add_argument("--n", type=int, default=4, help="Number of previews to generate")
    ap.add_argument("--out-dir", type=Path, default=Path("eval/uwf700_uveitis_label_transfer_previews"))
    # COCO boxes are in the coordinate system of `images[*].file_name`, which is the post-processed global image.
    args = ap.parse_args()

    coco = _read_json(args.coco)
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}

    by_img = {}
    for a in anns:
        by_img.setdefault(int(a["image_id"]), []).append(a)

    # Prefer images with the most annotations for visibility.
    ranked = sorted(images, key=lambda im: len(by_img.get(int(im["id"]), [])), reverse=True)
    ranked = [im for im in ranked if len(by_img.get(int(im["id"]), [])) > 0][: max(1, int(args.n))]

    out_dir = args.out_dir / f"{args.split_name}"
    _ensure_dir(out_dir)

    stats = []
    for im in ranked[: args.n]:
        coco_img_id = int(im["id"])
        image_id = str(im.get("image_id") or "")
        stem = image_id.split("::", 1)[-1]
        raw_path = Path("datasets/uwf-700/Images/Uveitis") / f"{stem}.jpg"
        lbl_path = Path("datasets/uwf-700/Labels/Uveitis") / f"{stem}.txt"

        preproc_path = Path(im["file_name"])

        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        pre = cv2.imread(str(preproc_path), cv2.IMREAD_COLOR)
        if raw is None or pre is None:
            continue

        raw_view = _draw_raw_polys(raw, _parse_uwf_obb(lbl_path))
        anns_img = by_img.get(coco_img_id, [])
        pre_view = _draw_coco_boxes(pre, anns_img, cat_id_to_name)

        raw_view = _fit_h(raw_view, 520)
        pre_view = _fit_h(pre_view, 520)

        divider = np.full((raw_view.shape[0], 10, 3), 255, dtype=np.uint8)
        combo = np.concatenate([raw_view, divider, pre_view], axis=1)
        cv2.putText(combo, f"{stem}  anns={len(anns_img)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)

        out_path = out_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), combo)

        ratios = [_nonblack_ratio(pre, a["bbox"]) for a in anns_img]
        stats.append((stem, len(anns_img), float(np.mean(ratios)) if ratios else 0.0))

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {len(stats)} previews to {out_dir}")
    if stats:
        avg = float(np.mean([s[2] for s in stats]))
        print(f"Avg non-black ratio inside boxes: {avg:.3f}")


if __name__ == "__main__":
    main()
