#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse
from pathlib import Path

from uveitis_pipeline.common import load_yaml
from uveitis_pipeline.manifest import (
    build_manifests,
    summarize_manifest,
    write_manifests,
    write_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified manifests + split file")
    parser.add_argument("--config", default="configs/stage0_manifest.yaml")
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.fold is not None:
        cfg["fold"] = args.fold

    manifests, split_dict = build_manifests(cfg)

    manifest_dir = Path(cfg["output"]["manifest_dir"])
    split_dir = Path(cfg["output"]["split_dir"])
    split_name = f"{cfg['output']['exp_name']}_{cfg.get('fold', 0)}.json"

    write_manifests(manifests, manifest_dir)
    write_splits(split_dict, split_dir / split_name)

    print(f"Wrote manifests to {manifest_dir}")
    print(f"Wrote split file to {split_dir / split_name}")

    for name, rows in manifests.items():
        summary = summarize_manifest(rows)
        print(f"[{name}] n_images={summary['n_images']}")
        print(f"  split_counts={summary['split_counts']}")
        print(f"  label_formats={summary['label_format_counts']}")
        if summary["class_counts"]:
            print(f"  class_counts={summary['class_counts']}")


if __name__ == "__main__":
    main()
