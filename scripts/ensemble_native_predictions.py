#!/usr/bin/env python3
"""Merge two native prediction files with a class-wise source map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uveitis_pipeline.common import read_jsonl, save_json, save_jsonl


def main() -> None:
    """Build merged predictions where each class is taken from source A or B."""
    parser = argparse.ArgumentParser(description="Class-wise ensemble for native predictions")
    parser.add_argument("--pred-a", required=True)
    parser.add_argument("--pred-b", required=True)
    parser.add_argument("--class-source-json", required=True, help='JSON map: {"class_name": "A"|"B"}')
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-meta", default="")
    args = parser.parse_args()

    rows_a = read_jsonl(args.pred_a)
    rows_b = read_jsonl(args.pred_b)
    map_a = {str(r.get("record_id", "")): r for r in rows_a}
    map_b = {str(r.get("record_id", "")): r for r in rows_b}
    source = {str(k): str(v).upper() for k, v in json.loads(Path(args.class_source_json).read_text(encoding="utf-8")).items()}

    out = []
    for rec_id in sorted(set(map_a.keys()) | set(map_b.keys())):
        ra = map_a.get(rec_id, {"record_id": rec_id, "predictions": []})
        rb = map_b.get(rec_id, {"record_id": rec_id, "predictions": []})

        preds = []
        for p in ra.get("predictions", []):
            cls = str(p.get("class_name", ""))
            if source.get(cls, "A") == "A":
                preds.append(p)
        for p in rb.get("predictions", []):
            cls = str(p.get("class_name", ""))
            if source.get(cls, "A") == "B":
                preds.append(p)

        out.append(
            {
                "record_id": rec_id,
                "image_id": ra.get("image_id", rb.get("image_id", "")),
                "image_path": ra.get("image_path", rb.get("image_path", "")),
                "predictions": preds,
            }
        )

    save_jsonl(args.out_jsonl, out)
    if args.out_meta:
        save_json(args.out_meta, {"pred_a": args.pred_a, "pred_b": args.pred_b, "class_source": source})


if __name__ == "__main__":
    main()
