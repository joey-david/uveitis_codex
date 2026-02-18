# `qa_yolo_obb_uwf_previews.py`

## Purpose
Quick qualitative QA for YOLOv8-OBB on UWF images:
- run inference tile-by-tile (crop from `preproc/global_1024` using `preproc/tiles_meta`)
- project OBBs back to global coordinates
- run an additional **global polygon NMS** across all tiles to reduce seam duplicates
- write overlay PNGs to `eval/`

## CLI
```bash
python scripts/qa_yolo_obb_uwf_previews.py \
  --weights out/weights/yolo_obb_main9_best.pt \
  --split val \
  --n 12 \
  --conf 0.05 \
  --tile-iou 0.7 \
  --global-nms-iou 0.3 \
  --out-dir eval/yolo_obb_previews
```

Pick images from a COCO file (stable list) instead of `splits/*`:
```bash
python scripts/qa_yolo_obb_uwf_previews.py \
  --weights out/weights/yolo_obb_main9_best.pt \
  --coco labels_coco/uwf700_val.json \
  --n 12
```

## Reads
- `preproc/global_1024/<image_id>.png`
- `preproc/tiles_meta/<image_id>.json`
- YOLO weights (`.pt`)

## Writes
- `eval/yolo_obb_previews/<weights_stem>_<split>_.../*.png`
  - filenames include `__preds{kept}_raw{raw}` (after vs before global NMS)

