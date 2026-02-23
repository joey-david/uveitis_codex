#!/usr/bin/env python3
"""Copy a YOLO dataset while keeping only selected class ids in labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_yaml(path: Path) -> dict:
    """Read YAML config."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve(base: Path, value: str) -> Path:
    """Resolve absolute path for a split directory."""
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def _label_path(image_path: Path) -> Path:
    """Map image path to label path."""
    return Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")


def _symlink(src: Path, dst: Path) -> None:
    """Create or replace symlink."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _filter_label(src: Path, dst: Path, keep: set[int], kept: Counter, dropped: Counter) -> None:
    """Write filtered label file keeping only classes in `keep`."""
    rows: list[str] = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            row = line.strip()
            if not row:
                continue
            cls = int(float(row.split()[0]))
            if cls in keep:
                rows.append(row)
                kept[cls] += 1
            else:
                dropped[cls] += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Filter YOLO labels to selected classes.")
    ap.add_argument("--src-data", type=Path, required=True, help="Source data.yaml")
    ap.add_argument("--out-dir", type=Path, required=True, help="Destination dataset directory")
    ap.add_argument("--keep", type=int, nargs="+", required=True, help="Class ids to keep")
    args = ap.parse_args()

    src_data = args.src_data.resolve()
    cfg = _load_yaml(src_data)
    base = src_data.parent
    keep = set(args.keep)

    names = cfg.get("names", {})
    if isinstance(names, dict):
        name_map = {int(k): str(v) for k, v in names.items()}
    else:
        name_map = {i: str(v) for i, v in enumerate(names)}

    out = args.out_dir.resolve()
    out_data = {
        "path": out.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {k: name_map[k] for k in sorted(name_map)},
    }
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    kept = Counter()
    dropped = Counter()
    split_image_counts: dict[str, int] = {}
    split_nonempty_labels: dict[str, int] = {}

    for split in ("train", "val"):
        src_img_dir = _resolve(base, cfg[split])
        out_img_dir = out / "images" / split
        out_lbl_dir = out / "labels" / split
        nonempty = 0
        count = 0
        for img in sorted(src_img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            count += 1
            _symlink(img.resolve(), out_img_dir / img.name)
            src_lbl = _label_path(img)
            dst_lbl = out_lbl_dir / f"{img.stem}.txt"
            _filter_label(src_lbl, dst_lbl, keep, kept, dropped)
            if dst_lbl.read_text(encoding="utf-8").strip():
                nonempty += 1
        split_image_counts[split] = count
        split_nonempty_labels[split] = nonempty

    (out / "data.yaml").write_text(yaml.safe_dump(out_data, sort_keys=False), encoding="utf-8")

    summary = {
        "src_data": src_data.as_posix(),
        "out_data": (out / "data.yaml").as_posix(),
        "keep_ids": sorted(keep),
        "keep_names": [name_map[k] for k in sorted(keep) if k in name_map],
        "kept_objects": {name_map.get(k, str(k)): int(v) for k, v in sorted(kept.items())},
        "dropped_objects": {name_map.get(k, str(k)): int(v) for k, v in sorted(dropped.items())},
        "split_image_counts": split_image_counts,
        "split_nonempty_label_files": split_nonempty_labels,
    }
    (out / "filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
