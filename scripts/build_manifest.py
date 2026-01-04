#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from stage1.data import find_images, make_relative, write_jsonl


def build_unlabeled(roots, output_path, rel_to):
    paths = find_images(roots)
    records = [{"path": make_relative(p, rel_to)} for p in sorted(paths)]
    write_jsonl(records, output_path)
    return len(records)


def build_uwf700(images_dir, out_dir, rel_to, train_frac, val_frac, seed):
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = sorted([p.name for p in images_dir.iterdir() if p.is_dir()])
    label_map = {name: idx for idx, name in enumerate(class_names)}
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for class_name in class_names:
        class_dir = images_dir / class_name
        paths = find_images([class_dir])
        rng.shuffle(paths)
        n_total = len(paths)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)
        split_paths = {
            "train": paths[:n_train],
            "val": paths[n_train : n_train + n_val],
            "test": paths[n_train + n_val :],
        }
        for split_name, split_list in split_paths.items():
            for path in split_list:
                splits[split_name].append(
                    {
                        "path": make_relative(path, rel_to),
                        "label": label_map[class_name],
                        "label_name": class_name,
                    }
                )
    for split_name, records in splits.items():
        write_jsonl(records, out_dir / f"{split_name}.jsonl")
    with (out_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    return {name: len(records) for name, records in splits.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Build image manifests for Stage 1")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    unlabeled = subparsers.add_parser("unlabeled", help="Build unlabeled manifest")
    unlabeled.add_argument("--roots", nargs="+", required=True, help="Image roots to scan")
    unlabeled.add_argument("--output", required=True, help="Output JSONL path")
    unlabeled.add_argument("--rel-to", default=Path.cwd(), help="Store paths relative to this root")

    uwf700 = subparsers.add_parser("uwf700", help="Build UWF-700 train/val/test manifests")
    uwf700.add_argument(
        "--images-dir",
        default="datasets/uwf-700/Images",
        help="UWF-700 Images directory",
    )
    uwf700.add_argument("--out-dir", default="manifests/uwf700", help="Output directory")
    uwf700.add_argument("--rel-to", default=Path.cwd(), help="Store paths relative to this root")
    uwf700.add_argument("--train-frac", type=float, default=0.8)
    uwf700.add_argument("--val-frac", type=float, default=0.1)
    uwf700.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "unlabeled":
        count = build_unlabeled(args.roots, args.output, args.rel_to)
        print(f"Wrote {count} images to {args.output}")
    else:
        counts = build_uwf700(
            args.images_dir,
            args.out_dir,
            args.rel_to,
            args.train_frac,
            args.val_frac,
            args.seed,
        )
        print(
            "Wrote UWF-700 manifests: "
            + ", ".join([f"{k}={v}" for k, v in counts.items()])
            + f" -> {args.out_dir}"
        )


if __name__ == "__main__":
    main()
