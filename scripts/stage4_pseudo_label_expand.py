#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse
import json
from pathlib import Path


def _load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_preds(pred_dir: Path, min_score: float) -> dict[str, list[dict]]:
    out = {}
    for p in sorted(pred_dir.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        preds = [x for x in data.get("predictions", []) if float(x.get("score", 0.0)) >= min_score]
        out[data.get("image_id", p.stem.replace("__", "::"))] = preds
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create pseudo-label COCO from predictions")
    parser.add_argument("--base-coco", required=True)
    parser.add_argument("--unlabeled-coco", default=None)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--min-score", type=float, default=0.9)
    parser.add_argument("--out-coco", required=True)
    args = parser.parse_args()

    base = _load_coco(Path(args.base_coco))
    unlabeled = _load_coco(Path(args.unlabeled_coco)) if args.unlabeled_coco else {"images": [], "annotations": []}
    preds = _collect_preds(Path(args.pred_dir), args.min_score)

    out = {
        "images": list(base["images"]),
        "annotations": list(base["annotations"]),
        "categories": list(base["categories"]),
    }
    img_lookup = {img.get("image_id", str(img["id"])): img for img in base["images"]}

    next_img_id = max((img["id"] for img in out["images"]), default=0) + 1
    next_ann_id = max((ann["id"] for ann in out["annotations"]), default=0) + 1

    for img in unlabeled.get("images", []):
        image_key = img.get("image_id", str(img["id"]))
        if image_key in img_lookup:
            continue
        if image_key not in preds:
            continue
        cloned = dict(img)
        cloned["id"] = next_img_id
        next_img_id += 1
        out["images"].append(cloned)
        img_lookup[image_key] = cloned

    for image_key, pred_list in preds.items():
        img = img_lookup.get(image_key)
        if img is None:
            continue
        for pred in pred_list:
            x1, y1, x2, y2 = pred["box"]
            out["annotations"].append(
                {
                    "id": next_ann_id,
                    "image_id": img["id"],
                    "category_id": int(pred["label"]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                    "pseudo": True,
                    "score": float(pred["score"]),
                }
            )
            next_ann_id += 1

    out_path = Path(args.out_coco)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(
        {
            "images": len(out["images"]),
            "annotations": len(out["annotations"]),
            "pseudo_annotations": sum(1 for a in out["annotations"] if a.get("pseudo")),
            "out": out_path.as_posix(),
        }
    )


if __name__ == "__main__":
    main()
