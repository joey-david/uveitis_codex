# `train_detector.py`

## Purpose
Train Faster R-CNN from a YAML config and produce checkpoints + metrics.

## CLI
```bash
python scripts/train_detector.py --config configs/train_uveitis_ft.yaml
```

## Reads
- Training YAML (`run`, `data`, `model`, `training`).
- COCO train/val files referenced by config.

## Writes
- `runs/<name>/config.yaml`
- `runs/<name>/checkpoints/*.pth`
- `runs/<name>/metrics.jsonl`
- `runs/<name>/val_report.json`

## Functions
| Function | Description |
|---|---|
| `main()` | Loads training config, calls training API, prints final best report. |

## Core module dependencies
- [`uveitis_pipeline.train`](../api/train.md)
- [`uveitis_pipeline.modeling`](../api/modeling.md)
