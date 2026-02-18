# `merge_yolo_datasets.py`

## Purpose
Merge multiple YOLO dataset roots into one by symlinking `images/*` and `labels/*`.

Use this when you want a single `data.yaml` and a single directory tree for training.

## CLI
```bash
python scripts/merge_yolo_datasets.py \
  --src out/yolo_obb/uwf700_tiles_main9 out/yolo_obb/fgadr_tiles_main9 \
  --out out/yolo_obb/merged_main9 \
  --splits train val
```

## Reads
- One or more YOLO dataset roots.

## Writes
- New YOLO root at `--out` containing symlinked files and a patched `data.yaml`.

