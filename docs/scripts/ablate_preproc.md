# `ablate_preproc.py`

## Purpose
Compare prediction directories (e.g., with/without preprocessing) by prediction count delta.

## CLI
```bash
python scripts/ablate_preproc.py --pred-a preds/exp_a --pred-b preds/exp_b --out eval/ablate_preproc.json
```

## Reads
- Two prediction directories with JSON outputs.

## Writes
- JSON summary at `--out`.

## Functions
| Function | Description |
|---|---|
| `main()` | Parses args, runs ablation summary, prints result. |

## Core module dependencies
- [`uveitis_pipeline.reports`](../api/reports.md)
