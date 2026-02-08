# `report_dataset.py`

## Purpose
Generate dataset-level and COCO-level count/statistics report.

## CLI
```bash
python scripts/report_dataset.py --manifests manifests/uwf700.jsonl manifests/fgadr.jsonl --cocos labels_coco/uwf700_train.json --out eval/report_dataset.json
```

## Reads
- Manifest JSONL files.
- Optional COCO files.

## Writes
- JSON report at `--out`.

## Functions
| Function | Description |
|---|---|
| `main()` | Parses args, runs dataset report, prints output dictionary. |

## Core module dependencies
- [`uveitis_pipeline.reports`](../api/reports.md)
