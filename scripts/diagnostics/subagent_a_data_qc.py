#!/usr/bin/env python3
"""Data and label integrity checks for YOLO-OBB datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


def _poly_area(xs: list[float], ys: list[float]) -> float:
    """Return polygon area via shoelace formula."""
    n = len(xs)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) * 0.5


def _global_id_from_stem(stem: str) -> str:
    """Map tile filename stem to global image id."""
    return stem.split("__tile_")[0]


def _read_data_yaml(path: Path) -> dict:
    """Load a YOLO data.yaml file."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _scan_split(labels_dir: Path, num_classes: int) -> dict:
    """Scan one split labels directory and emit integrity stats."""
    cls_hist: Counter[int] = Counter()
    invalid_dim = 0
    invalid_class = 0
    coord_oob = 0
    degenerate = 0
    empty_files = 0
    ann_count = 0
    tile_files = sorted(labels_dir.glob("*.txt"))
    global_ids = set()

    for p in tile_files:
        global_ids.add(_global_id_from_stem(p.stem))
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            empty_files += 1
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) != 9:
                invalid_dim += 1
                continue
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            xs = coords[0::2]
            ys = coords[1::2]
            ann_count += 1
            if cls < 0 or cls >= num_classes:
                invalid_class += 1
            else:
                cls_hist[cls] += 1
            if any((v < -1e-6 or v > 1.000001) for v in coords):
                coord_oob += 1
            if _poly_area(xs, ys) < 1e-6:
                degenerate += 1

    return {
        "num_tiles": len(tile_files),
        "num_global_images": len(global_ids),
        "num_annotations": ann_count,
        "empty_label_files": empty_files,
        "invalid_line_dim": invalid_dim,
        "invalid_class": invalid_class,
        "coord_out_of_bounds": coord_oob,
        "degenerate_polygons": degenerate,
        "class_hist": {str(k): int(v) for k, v in sorted(cls_hist.items())},
        "global_ids": sorted(global_ids),
    }


def run_dataset_qc(data_yaml: Path) -> dict:
    """Run integrity checks for train/val splits of one dataset."""
    cfg = _read_data_yaml(data_yaml)
    root = Path(cfg["path"])
    names = cfg.get("names", {})
    if isinstance(names, dict):
        num_classes = len(names)
    elif isinstance(names, list):
        num_classes = len(names)
    else:
        raise ValueError(f"Unexpected names format in {data_yaml}")

    train_labels = root / str(cfg["train"]).replace("images", "labels")
    val_labels = root / str(cfg["val"]).replace("images", "labels")
    train_stats = _scan_split(train_labels, num_classes)
    val_stats = _scan_split(val_labels, num_classes)
    train_globals = set(train_stats.pop("global_ids"))
    val_globals = set(val_stats.pop("global_ids"))
    leakage = sorted(train_globals & val_globals)

    return {
        "data_yaml": data_yaml.as_posix(),
        "dataset_root": root.as_posix(),
        "num_classes": num_classes,
        "train": train_stats,
        "val": val_stats,
        "split_leakage_global_ids": leakage,
        "split_leakage_count": len(leakage),
    }


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="YOLO-OBB dataset integrity checks.")
    ap.add_argument("--data-yaml", type=Path, action="append", required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    report = {"datasets": [run_dataset_qc(p) for p in args.data_yaml]}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
