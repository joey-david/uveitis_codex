# `report_preproc.py`

## Purpose
Generate preprocessing quality report with before/after grids and channel histograms.

## CLI
```bash
python scripts/report_preproc.py --manifest manifests/uwf700.jsonl --preproc-root preproc --out-dir eval/preproc [--sample-n 24]
```

## Reads
- Manifest entries and corresponding preprocessed files.

## Writes
- `preproc_report.json`
- `raw_vs_norm_grid.png`
- `roi_hist_before_after.png`

## Functions
| Function | Description |
|---|---|
| `main()` | Parses args, runs preprocessing report, prints summary. |

## Core module dependencies
- [`uveitis_pipeline.reports`](../api/reports.md)
