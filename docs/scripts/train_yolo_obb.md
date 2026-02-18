# `train_yolo_obb.py`

## Purpose
Train an Ultralytics YOLOv8-OBB model using a YOLO dataset produced by `export_yolo_obb.py`.

## CLI
```bash
python scripts/train_yolo_obb.py \
  --model yolov8m-obb.pt \
  --data out/yolo_obb/uwf700_tiles_main9/data.yaml \
  --imgsz 1280 \
  --epochs 30 \
  --batch 8 \
  --name uwf700_tiles_main9
```

## Reads
- YOLO `data.yaml` + `images/*` and `labels/*`.
- Optional pretrained model weights (`--model`).

## Writes
- Ultralytics run folder under `runs/yolo_obb/<name>/...` (checkpoints, logs).

