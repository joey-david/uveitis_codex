#!/usr/bin/env python3
"""Build native fine-grained labels (polygons/OBB) from masks and OBB annotations."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

from uveitis_pipeline.common import load_yaml, read_jsonl, save_json
from uveitis_pipeline.labels_native import build_native_labels_from_manifest, filter_class_map


def _select_allowed_classes(label_space_cfg: dict, mode: str) -> set[str]:
    """Select active class names according to the requested label-space mode."""
    if mode == "all":
        classes = set(label_space_cfg.get("main_detector", {}).get("class_names", []))
        vas = label_space_cfg.get("vascularite", {}).get("class_name")
        if vas:
            classes.add(str(vas))
        return classes
    if mode == "vascularite":
        vas = label_space_cfg.get("vascularite", {}).get("class_name", "vascularite")
        return {str(vas)}
    return set(label_space_cfg.get("main_detector", {}).get("class_names", []))


def main() -> None:
    """Run native-label export for requested datasets/splits."""
    parser = argparse.ArgumentParser(description="Build native labels from stage-0 preprocessed assets")
    parser.add_argument("--config", default="configs/stage0_labels.yaml")
    parser.add_argument(
        "--mode",
        choices=["main", "vascularite", "all"],
        default=None,
        help="Override label-space mode. If omitted, config input.label_space_mode is used.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rows = []
    for manifest in cfg["input"]["manifests"]:
        rows.extend(read_jsonl(manifest))

    split = json.loads(Path(cfg["input"]["split_json"]).read_text(encoding="utf-8"))
    class_map = load_yaml(cfg["input"]["class_map_yaml"])

    label_space_path = cfg["input"].get("label_space_yaml")
    mode = args.mode or cfg["input"].get("label_space_mode", "main")
    if label_space_path:
        label_space = load_yaml(label_space_path)
        allowed = _select_allowed_classes(label_space, str(mode))
        class_map = filter_class_map(class_map, allowed)

    preproc_root = Path(cfg["input"]["preproc_root"])
    labels_root = Path(cfg["output"].get("labels_root") or cfg["output"].get("labels_dir") or "labels_native")
    labels_root.mkdir(parents=True, exist_ok=True)

    save_json(labels_root / "class_map_active.json", class_map)

    target_datasets = cfg["input"].get("target_datasets", ["fgadr", "uwf700"])
    target_splits = cfg["input"].get("splits", ["train", "val", "test"])

    include_global = bool(cfg["build"].get("enable_global", True))
    include_tiles = bool(cfg["build"].get("enable_tiles", True))
    min_comp_area = int(cfg["build"].get("min_component_area", 10))
    min_tile_obj_ratio = float(cfg["build"].get("min_tile_obj_ratio", cfg["build"].get("min_tile_box_ratio", 0.15)))
    simplify_eps = float(cfg["build"].get("polygon_simplify_eps", 1.25))

    summary: dict[str, dict] = {}
    for dataset in target_datasets:
        ds_rows = [r for r in rows if r.get("dataset") == dataset]
        if not ds_rows:
            continue
        for split_name in target_splits:
            split_ids = set(split.get(split_name, []))
            stats = build_native_labels_from_manifest(
                manifest_rows=ds_rows,
                split_ids=split_ids,
                class_map_cfg=class_map,
                preproc_root=preproc_root,
                out_root=labels_root,
                dataset_name=dataset,
                split_name=split_name,
                include_global=include_global,
                include_tiles=include_tiles,
                min_comp_area=min_comp_area,
                min_tile_obj_ratio=min_tile_obj_ratio,
                simplify_eps=simplify_eps,
            )
            key = f"{dataset}_{split_name}"
            summary[key] = stats
            print(f"[{key}] {stats}")

    save_json(labels_root / "summary.json", summary)
    print(f"Saved native labels to {labels_root}")


if __name__ == "__main__":
    main()
