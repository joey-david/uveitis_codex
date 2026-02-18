# `make_yolo_mix.py`

## Purpose
Create a new YOLO dataset by mixing multiple YOLO dataset roots via symlinks.

This is mainly used to upsample a small dataset (e.g. UWF uveitis) when training YOLO.

## CLI
```bash
python scripts/make_yolo_mix.py \
  --train-src out/yolo_obb/uwf700_tiles_main9 \
  --val-src out/yolo_obb/uwf700_tiles_main9 \
  --train-repeat out/yolo_obb/uwf700_tiles_main9=10 \
  --out out/yolo_obb/mix_main9_x10
```

## Reads
- One or more YOLO roots (each has `data.yaml`, `images/<split>`, `labels/<split>`).

## Writes
- New YOLO root at `--out` containing symlinked files.

