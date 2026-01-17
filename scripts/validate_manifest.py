#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mvcavit.data import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Validate multi-view manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve(path, root):
    path = Path(path)
    if root and not path.is_absolute():
        path = Path(root) / path
    return path


def main():
    args = parse_args()
    root = Path(args.root) if args.root else None
    errors = []
    count = 0
    for rec in read_jsonl(args.manifest):
        macula = resolve(rec.get("macula_path") or rec.get("macula") or rec.get("path"), root)
        optic = resolve(rec.get("optic_path") or rec.get("optic") or macula, root)
        if not macula.exists():
            errors.append({"error": "missing_macula", "path": str(macula)})
        if not optic.exists():
            errors.append({"error": "missing_optic", "path": str(optic)})
        boxes = rec.get("boxes", [])
        if boxes:
            with Image.open(macula) as img:
                w, h = img.size
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    errors.append({"error": "box_out_of_bounds", "box": box, "path": str(macula)})
                    break
        count += 1
        if args.limit and count >= args.limit:
            break
    print(json.dumps({"records": count, "errors": errors[:20], "error_count": len(errors)}))


if __name__ == "__main__":
    main()
