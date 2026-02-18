# `filter_yolo_dataset.py`

## Purpose
Filter a YOLO dataset split into:
- `pos`: images that have non-empty label files
- `bg`: images that have empty label files

This is useful for balancing background-only vs positive samples.

## CLI
Keep positives only:
```bash
python scripts/filter_yolo_dataset.py \
  --src out/yolo_obb/uwf700_tiles_main9 \
  --out out/yolo_obb/uwf700_tiles_main9_pos \
  --split train \
  --mode pos
```

Keep background-only:
```bash
python scripts/filter_yolo_dataset.py \
  --src out/yolo_obb/uwf700_tiles_main9 \
  --out out/yolo_obb/uwf700_tiles_main9_bg \
  --split train \
  --mode bg
```

## Reads
- YOLO dataset root.

## Writes
- Filtered YOLO dataset root at `--out` (symlinks).

