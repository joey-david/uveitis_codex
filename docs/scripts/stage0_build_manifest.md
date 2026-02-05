# `stage0_build_manifest.py`

## Purpose
Build per-dataset manifests and a train/val/test split file.

## CLI
```bash
python scripts/stage0_build_manifest.py --config configs/stage0_manifest.yaml [--fold N]
```

## Reads
- Dataset roots from config (`datasets.*.root`).

## Writes
- `manifests/<dataset>.jsonl`
- `manifests/<dataset>.csv`
- `splits/<exp_name>_<fold>.json`

## Functions
| Function | Description |
|---|---|
| `main()` | Parses args, loads YAML, builds manifests/splits, writes outputs, prints summaries. |

## Core module dependencies
- [`uveitis_pipeline.manifest`](../api/manifest.md)
- [`uveitis_pipeline.common`](../api/common.md)
