# Perf Sweep: Main Detector (Main9) + Vascularite Module

## Active Label Space
- Main detector classes: `exudats`, `hemorragie`, `macroanevrisme_arteriel`, `ischemie_retine`, `nodule_choroidien`,
  `hyalite`, `foyer_choroidien`, `granulome_choroidien`, `oedeme_papillaire`
- Separate model: `vascularite`

Source of truth:
- `configs/active_label_space.yaml`
- `configs/main_detector_classes.txt`

## Main9 (UWF-only)

### uwf_main9_y8l_1280_adamw_e50 (stopped early)
- Train/val: `out/yolo_obb/uwf700_tiles_main9/data.yaml` (val=UWF tiles)
- Model: `yolov8l-obb.pt`
- Args: AdamW + cosine LR, `imgsz=1280`, `batch=8`
- Run: `runs/obb/runs/yolo_obb/uwf_main9_y8l_1280_adamw_e50/`
- Best observed (at stop): mAP50(B) ~= `0.189` (epoch 16)

### uwf_main9_y8l_1536_adamw_e30_ft (finetune)
- Train/val: `out/yolo_obb/uwf700_tiles_main9/data.yaml` (val=UWF tiles)
- Model init: `runs/obb/runs/yolo_obb/uwf_main9_y8l_1280_adamw_e50/weights/best.pt`
- Args: AdamW + cosine LR, `imgsz=1536`, `batch=4`, `epochs=30`
- Run: `runs/obb/runs/yolo_obb/uwf_main9_y8l_1536_adamw_e30_ft/`
- Best observed: epoch 2
  - mAP50(B)=`0.21964`
  - mAP50-95(B)=`0.13692`

## Qualitative Previews
- Combined overlays (main + vascularite) on `preproc/global_1024`:
  - `eval/yolo_obb_previews_combined/best__plus__best__n6/`

## Vascularite (separate)

### uwf_vascularite_y8l_1536_adamw_e20 (stopped early)
- Train/val: `out/yolo_obb/uwf700_tiles_vascularite_only/data.yaml`
- Model: `yolov8l-obb.pt`
- Args: AdamW + cosine LR, `imgsz=1536`, `batch=4`
- Run: `runs/obb/runs/yolo_obb/uwf_vascularite_y8l_1536_adamw_e20/`
- Best observed (at stop): epoch 14
  - mAP50(B)=`0.19029`
  - mAP50-95(B)=`0.05167`

Updated combined previews using these weights:
- `eval/yolo_obb_previews_combined/main9_vasc1536/best__plus__best__n6/`
