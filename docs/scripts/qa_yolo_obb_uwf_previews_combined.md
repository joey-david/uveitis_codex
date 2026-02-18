# `qa_yolo_obb_uwf_previews_combined.py`

## Purpose
Generate overlay previews that combine:
- main detector YOLOv8-OBB model (active Main9 label space)
- vascularite YOLOv8-OBB model

It runs per-tile inference for each model, projects to global coordinates, applies global polygon NMS per model, then draws both on the same global image.

## CLI
```bash
python scripts/qa_yolo_obb_uwf_previews_combined.py \
  --weights-main out/weights/yolo_obb_main9_best.pt \
  --weights-vascularite out/weights/yolo_obb_vascularite_best.pt \
  --split val \
  --n 12 \
  --conf-main 0.05 \
  --conf-vascularite 0.05 \
  --global-nms-iou 0.3 \
  --out-dir eval/yolo_obb_previews_combined
```

## Reads
- `preproc/global_1024/<image_id>.png`
- `preproc/tiles_meta/<image_id>.json`
- YOLO weights (`.pt`) for main and vascularite models

## Writes
- `eval/yolo_obb_previews_combined/<main_stem>__plus__<vascularite_stem>_.../*.png`

