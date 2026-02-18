# `export_yolo_obb.py`

## Purpose
Export our COCO(+`obb` field) labels to a YOLOv8-OBB dataset folder with `images/{train,val}` and `labels/{train,val}`.

This is useful for fast iteration with Ultralytics YOLO OBB while keeping Stage 0/labels as the source of truth.

## CLI
Export a main-detector dataset (active label space, excluding `vascularite`):
```bash
python scripts/export_yolo_obb.py \
  --coco-train labels_coco/uwf700_train_tiles.json \
  --coco-val labels_coco/uwf700_val_tiles.json \
  --out out/yolo_obb/uwf700_tiles_main9 \
  --keep-file configs/main_detector_classes.txt
```

Export a vascularite-only dataset:
```bash
python scripts/export_yolo_obb.py \
  --coco-train labels_coco/fgadr_train_tiles.json \
  --coco-val labels_coco/fgadr_val_tiles.json \
  --out out/yolo_obb/fgadr_tiles_vascularite \
  --keep-name vascularite
```

## Reads
- COCO labels JSON from Stage 0.4 (`labels_coco/*_tiles.json`).
- Optional keep/drop lists (`--keep-*`, `--drop-name`).

## Writes
- `out/yolo_obb/<name>/data.yaml`
- `out/yolo_obb/<name>/images/{train,val}/*` (symlinks by default)
- `out/yolo_obb/<name>/labels/{train,val}/*.txt`

## Notes
- By default, images are symlinked (smaller/faster). Use `--copy` only if you need a fully materialized dataset.
- For UWF tiles, filenames are made unique by including the parent `image_id` and `tile_id`.

