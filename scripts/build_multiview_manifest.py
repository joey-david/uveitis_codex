#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mvcavit.data import find_images, make_relative, write_jsonl


def _load_labels(path):
    if not path:
        return {}
    labels = {}
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["key"]] = int(row["label"])
    return labels


def build_from_pairs(root, macula_suffix, optic_suffix, output, rel_to, default_label, labels_csv):
    root = Path(root)
    labels = _load_labels(labels_csv)
    pairs = {}
    for path in find_images([root]):
        stem = path.stem
        if stem.endswith(macula_suffix):
            key = stem[: -len(macula_suffix)]
            pairs.setdefault(key, {})["macula_path"] = path
        elif stem.endswith(optic_suffix):
            key = stem[: -len(optic_suffix)]
            pairs.setdefault(key, {})["optic_path"] = path
    records = []
    for key, pair in pairs.items():
        if "macula_path" not in pair or "optic_path" not in pair:
            continue
        records.append(
            {
                "macula_path": make_relative(pair["macula_path"], rel_to),
                "optic_path": make_relative(pair["optic_path"], rel_to),
                "label": labels.get(key, default_label),
            }
        )
    write_jsonl(records, output)
    return len(records)


def build_from_csv(csv_path, output, rel_to):
    records = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {
                "macula_path": make_relative(row.get("macula_path") or row.get("macula"), rel_to),
                "optic_path": make_relative(row.get("optic_path") or row.get("optic"), rel_to),
                "label": int(row.get("label", 0)),
            }
            if row.get("boxes"):
                rec["boxes"] = json.loads(row["boxes"])
            if row.get("macula_center"):
                rec["macula_center"] = json.loads(row["macula_center"])
            if row.get("optic_center"):
                rec["optic_center"] = json.loads(row["optic_center"])
            records.append(rec)
    write_jsonl(records, output)
    return len(records)


def parse_args():
    parser = argparse.ArgumentParser(description="Build multi-view manifests")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    pairs = subparsers.add_parser("pairs")
    pairs.add_argument("--root", required=True, help="Root directory of images")
    pairs.add_argument("--macula-suffix", default="_M", help="Suffix for macula images")
    pairs.add_argument("--optic-suffix", default="_O", help="Suffix for optic images")
    pairs.add_argument("--output", required=True, help="Output JSONL path")
    pairs.add_argument("--rel-to", default=Path.cwd(), help="Store paths relative to this root")
    pairs.add_argument("--default-label", type=int, default=0)
    pairs.add_argument("--labels-csv", help="CSV with columns: key,label")

    csv_mode = subparsers.add_parser("csv")
    csv_mode.add_argument("--csv", required=True, help="CSV with macula_path, optic_path, label, boxes")
    csv_mode.add_argument("--output", required=True, help="Output JSONL path")
    csv_mode.add_argument("--rel-to", default=Path.cwd(), help="Store paths relative to this root")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "pairs":
        count = build_from_pairs(
            args.root,
            args.macula_suffix,
            args.optic_suffix,
            args.output,
            args.rel_to,
            args.default_label,
            args.labels_csv,
        )
    else:
        count = build_from_csv(args.csv, args.output, args.rel_to)
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
