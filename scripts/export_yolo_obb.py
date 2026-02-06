#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _symlink_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if use_symlink:
        # Keep symlinks valid inside Docker: use repo-relative paths (not host-absolute).
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)
    else:
        shutil.copy2(src, dst)


def _write_yaml(path: Path, obj: dict) -> None:
    # Tiny YAML writer to avoid adding deps for a single file.
    lines: list[str] = []
    for k, v in obj.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_split(coco: dict, split: str, out_root: Path, use_symlink: bool) -> None:
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])
    cat_id_to_idx = {int(c["id"]): i for i, c in enumerate(sorted(cats, key=lambda c: int(c["id"])))}

    by_img: dict[int, list[dict]] = {}
    for a in anns:
        by_img.setdefault(int(a["image_id"]), []).append(a)

    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    _ensure_dir(img_dir)
    _ensure_dir(lbl_dir)

    n_img = 0
    n_ann = 0
    for im in images:
        img_id = int(im["id"])
        src = Path(im["file_name"])
        if not src.exists():
            continue
        # Tile filenames repeat (tile_000.png etc), so include image_id/tile_id for uniqueness.
        if im.get("tile_id") is not None and im.get("image_id"):
            stem = f"{str(im['image_id']).replace('::','__')}__{im['tile_id']}"
        else:
            stem = src.stem
        dst = img_dir / f"{stem}{src.suffix}"
        _symlink_or_copy(src, dst, use_symlink=use_symlink)

        lines: list[str] = []
        for a in by_img.get(img_id, []):
            cls = int(cat_id_to_idx[int(a["category_id"])])
            obb = a.get("obb")
            if isinstance(obb, list) and len(obb) == 8:
                vals = [float(v) for v in obb]
            else:
                # Fallback: represent an axis-aligned bbox as a 4-point polygon.
                x, y, w, h = [float(v) for v in a["bbox"]]
                iw = float(im["width"])
                ih = float(im["height"])
                x1, y1, x2, y2 = x, y, x + w, y + h
                vals = [x1 / iw, y1 / ih, x2 / iw, y1 / ih, x2 / iw, y2 / ih, x1 / iw, y2 / ih]

            lines.append(" ".join([str(cls)] + [f"{v:.6f}" for v in vals]))
            n_ann += 1
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_img += 1

    print(f"[{split}] images={n_img} obb_anns={n_ann}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export COCO(+obb field) to YOLOv8-OBB dataset structure.")
    ap.add_argument("--coco-train", type=Path, required=True)
    ap.add_argument("--coco-val", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out/yolo_obb/uwf700_global"))
    ap.add_argument("--copy", action="store_true", help="Copy images instead of symlinking (slower, bigger).")
    ap.add_argument(
        "--drop-name",
        action="append",
        default=[],
        help="Category name to drop (repeatable). Example: --drop-name vascularite",
    )
    ap.add_argument(
        "--keep-name",
        action="append",
        default=[],
        help="If set, keep only these category names (repeatable). Example: --keep-name vascularite",
    )
    ap.add_argument(
        "--keep-file",
        type=Path,
        default=None,
        help="Optional file with one category name per line (blank lines and '#' comments ignored).",
    )
    args = ap.parse_args()

    coco_train = _read_json(args.coco_train)
    coco_val = _read_json(args.coco_val)
    out_root = args.out
    _ensure_dir(out_root)

    keep = {str(x).strip() for x in args.keep_name if str(x).strip()}
    if args.keep_file:
        for line in args.keep_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            keep.add(s)
    drop = {str(x).strip() for x in args.drop_name if str(x).strip()}
    cats_all = sorted(coco_train.get("categories", []), key=lambda c: int(c["id"]))
    if keep:
        cats = [c for c in cats_all if str(c.get("name")) in keep]
    else:
        cats = [c for c in cats_all if str(c.get("name")) not in drop]
    keep_ids = {int(c["id"]) for c in cats}
    if keep:
        print(f"Keeping only categories: {sorted(keep)}")
    if drop and not keep:
        print(f"Dropping categories: {sorted(drop)}")
    if keep or drop:
        print(f"Keeping categories: {[str(c['name']) for c in cats]}")

    def _filter(coco: dict) -> dict:
        if not (keep or drop):
            return coco
        out = dict(coco)
        out["categories"] = [c for c in coco.get("categories", []) if int(c["id"]) in keep_ids]
        out["annotations"] = [a for a in coco.get("annotations", []) if int(a["category_id"]) in keep_ids]
        return out

    coco_train = _filter(coco_train)
    coco_val = _filter(coco_val)

    names = {i: str(c["name"]) for i, c in enumerate(sorted(coco_train.get("categories", []), key=lambda c: int(c["id"])))}
    data_yaml = {
        "path": out_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    _write_yaml(out_root / "data.yaml", data_yaml)

    _export_split(coco_train, "train", out_root, use_symlink=not args.copy)
    _export_split(coco_val, "val", out_root, use_symlink=not args.copy)
    print(f"Wrote {out_root/'data.yaml'}")


if __name__ == "__main__":
    main()
