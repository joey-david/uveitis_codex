# `report_training.py`

## Purpose
Summarize a training run and render training curves.

## CLI
```bash
python scripts/report_training.py --run-dir runs/uveitis_ft --out-json eval/report_training.json --out-png eval/training_curves.png
```

## Reads
- `metrics.jsonl`
- `val_report.json`

## Writes
- `--out-json`
- `--out-png`

## Functions
| Function | Description |
|---|---|
| `main()` | Parses args, runs training report generation, prints summary. |

## Core module dependencies
- [`uveitis_pipeline.reports`](../api/reports.md)
