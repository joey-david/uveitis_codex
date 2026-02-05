#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse
import json
from collections import defaultdict
from pathlib import Path

from uveitis_pipeline.common import load_yaml, read_jsonl, save_json
from uveitis_pipeline.labels import build_coco_from_manifest, summarize_coco


def main() -> None:
    parser = argparse.ArgumentParser(description="Build COCO labels from masks/OBBs")
    parser.add_argument("--config", default="configs/stage0_labels.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    rows = []
    for m in cfg["input"]["manifests"]:
        rows.extend(read_jsonl(m))

    split = json.loads(Path(cfg["input"]["split_json"]).read_text(encoding="utf-8"))
    class_map = load_yaml(cfg["input"]["class_map_yaml"])
    preproc_root = Path(cfg["input"]["preproc_root"])

    labels_dir = Path(cfg["output"]["labels_dir"])
    debug_dir = Path(cfg["output"]["debug_dir"])
    summary = {}

    target_datasets = cfg["input"].get("target_datasets", ["fgadr", "uwf700"])

    for dataset in target_datasets:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        if not ds_rows:
            continue

        for split_name in cfg["input"].get("splits", ["train", "val", "test"]):
            split_ids = set(split.get(split_name, []))

            coco_global = build_coco_from_manifest(
                ds_rows,
                split_ids,
                class_map,
                preproc_root,
                labels_dir / f"{dataset}_{split_name}.json",
                debug_dir / f"{dataset}_{split_name}",
                tile_mode=False,
                min_comp_area=int(cfg["build"].get("min_component_area", 8)),
            )
            coco_tiles = build_coco_from_manifest(
                ds_rows,
                split_ids,
                class_map,
                preproc_root,
                labels_dir / f"{dataset}_{split_name}_tiles.json",
                debug_dir / f"{dataset}_{split_name}_tiles",
                tile_mode=True,
                min_comp_area=int(cfg["build"].get("min_component_area", 8)),
                min_tile_box_ratio=float(cfg["build"].get("min_tile_box_ratio", 0.2)),
            )

            summary[f"{dataset}_{split_name}"] = {
                "global": summarize_coco(coco_global),
                "tiles": summarize_coco(coco_tiles),
            }
            print(f"[{dataset}/{split_name}] global={summary[f'{dataset}_{split_name}']['global']}")
            print(f"[{dataset}/{split_name}] tiles={summary[f'{dataset}_{split_name}']['tiles']}")

    save_json(labels_dir / "summary.json", summary)
    print(f"Saved COCO labels to {labels_dir}")


if __name__ == "__main__":
    main()
