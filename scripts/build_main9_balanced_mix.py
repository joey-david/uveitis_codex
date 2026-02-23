#!/usr/bin/env python3
"""Build a class-balanced YOLO-OBB dataset by repeating rare-class train tiles."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_yaml(path: Path) -> dict:
    """Read a YAML file."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_split(base: Path, split_path: str) -> Path:
    """Resolve a split path from YOLO data.yaml."""
    p = Path(split_path)
    return p if p.is_absolute() else (base / p).resolve()


def _label_for_image(image_path: Path) -> Path:
    """Get YOLO label path for an image path."""
    return Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")


def _classes_in_label(label_path: Path) -> list[int]:
    """Read class ids present in a YOLO label file."""
    if not label_path.exists():
        return []
    classes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row:
            continue
        classes.append(int(float(row.split()[0])))
    return classes


def _safe_symlink(src: Path, dst: Path) -> None:
    """Create a symlink, replacing pre-existing files."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _name_map(names: dict | list) -> dict[int, str]:
    """Normalize YOLO names field to int->name mapping."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Build class-balanced Main9 YOLO dataset.")
    ap.add_argument("--src-data", type=Path, required=True, help="Source YOLO data.yaml.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output dataset directory.")
    ap.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Target object count per class. Default: max class count in source train split.",
    )
    ap.add_argument("--max-repeat", type=int, default=8, help="Max repeats for a train tile.")
    ap.add_argument("--bg-repeat", type=int, default=1, help="Repeats for train background tiles.")
    args = ap.parse_args()

    src_data = args.src_data.resolve()
    src = _load_yaml(src_data)
    base = src_data.parent
    train_dir = _resolve_split(base, src["train"])
    val_dir = _resolve_split(base, src["val"])
    names = _name_map(src["names"])

    train_images = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    val_images = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    class_counts = Counter()
    per_image_classes: dict[Path, list[int]] = {}
    for img in train_images:
        cls = _classes_in_label(_label_for_image(img))
        per_image_classes[img] = cls
        class_counts.update(cls)

    if not class_counts:
        raise RuntimeError("No train labels found in source dataset.")

    target = args.target_count or max(class_counts.values())
    class_weight = {c: target / max(1, n) for c, n in class_counts.items()}

    out = args.out_dir.resolve()
    out_train_img = out / "images" / "train"
    out_train_lbl = out / "labels" / "train"
    out_val_img = out / "images" / "val"
    out_val_lbl = out / "labels" / "val"
    for p in (out_train_img, out_train_lbl, out_val_img, out_val_lbl):
        p.mkdir(parents=True, exist_ok=True)

    repeats = Counter()
    for img in train_images:
        cls = per_image_classes[img]
        if not cls:
            rep = args.bg_repeat
        else:
            rep = math.ceil(max(class_weight[c] for c in cls))
            rep = max(1, min(args.max_repeat, rep))
        repeats[rep] += 1

        label_src = _label_for_image(img).resolve()
        stem = img.stem
        suffix = img.suffix
        for i in range(rep):
            name = f"{stem}__r{i}{suffix}"
            dst_img = out_train_img / name
            dst_lbl = out_train_lbl / f"{stem}__r{i}.txt"
            _safe_symlink(img.resolve(), dst_img)
            _safe_symlink(label_src, dst_lbl)

    for img in val_images:
        label_src = _label_for_image(img).resolve()
        _safe_symlink(img.resolve(), out_val_img / img.name)
        _safe_symlink(label_src, out_val_lbl / f"{img.stem}.txt")

    data_out = {
        "path": out.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {int(k): v for k, v in sorted(names.items(), key=lambda kv: kv[0])},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_out, sort_keys=False), encoding="utf-8")

    balanced_counts = Counter()
    for img in train_images:
        rep = math.ceil(max((class_weight[c] for c in per_image_classes[img]), default=args.bg_repeat))
        rep = max(1, min(args.max_repeat, rep))
        for c in per_image_classes[img]:
            balanced_counts[c] += rep

    summary = {
        "src_data": src_data.as_posix(),
        "out_data": (out / "data.yaml").as_posix(),
        "target_count": target,
        "max_repeat": args.max_repeat,
        "bg_repeat": args.bg_repeat,
        "source_class_counts": {names.get(c, str(c)): int(class_counts[c]) for c in sorted(class_counts)},
        "balanced_object_proxy_counts": {names.get(c, str(c)): int(balanced_counts[c]) for c in sorted(balanced_counts)},
        "repeat_histogram": {int(k): int(v) for k, v in sorted(repeats.items())},
        "num_train_images_source": len(train_images),
        "num_train_images_balanced": int(sum(r * n for r, n in repeats.items())),
        "num_val_images": len(val_images),
    }
    (out / "balance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
