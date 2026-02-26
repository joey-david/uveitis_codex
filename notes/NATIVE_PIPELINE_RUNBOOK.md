# Native Pipeline Runbook

## 0) Build manifests/splits
```bash
python scripts/stage0_build_manifest.py --config configs/stage0_manifest.yaml
```
Expected:
- `manifests/*.jsonl`
- `splits/stage0_0.json`

## 1) Preprocess (SAM2 ROI -> ROI-only normalization -> global -> tiles)
```bash
python scripts/stage0_preprocess.py --config configs/stage0_preprocess.yaml
```
Expected under `preproc/`:
- `roi_masks/`, `crops/`, `norm/`, `global/`, `tiles/`, `tiles_meta/`, `verify/preprocess_metrics.json`

## 2) Build native fine-grained labels (no COCO)
Main detector classes:
```bash
python scripts/stage0_build_labels.py --config configs/stage0_labels_main9.yaml
```
Vascularite-only branch:
```bash
python scripts/stage0_build_labels.py --config configs/stage0_labels_vascularite.yaml
```
Expected:
- `labels_native/*.jsonl`
- `labels_native/records/**.json`
- `labels_native/class_map_active.json`

## 3) QA native labels
```bash
python scripts/qa_native_labels.py \
  --index labels_native/uwf700_val_global.jsonl \
  --out eval/native_labels_qa_val \
  --n 30
```
Expected:
- `eval/native_labels_qa_val/overlays/*.png`
- `eval/native_labels_qa_val/masks/*.png`
- `eval/native_labels_qa_val/qa_summary.json`

## 4) RETFound contrastive adaptation
```bash
python scripts/stage4_adapt_retfound.py --config configs/stage4_adapt_retfound.yaml
```
Expected:
- `runs/retfound_adapt/<run_name>/best.pt`
- `runs/retfound_adapt/<run_name>/metrics.json`

## 5) Mask-first training
```bash
python scripts/stage5_train_mask_head.py --config configs/stage5_train_mask_head.yaml
```
Expected:
- `runs/retfound_mask/<run_name>/best.pt`
- `runs/retfound_mask/<run_name>/metrics.json`

## 6) Inference (mask -> OBB/polygon)
```bash
python scripts/stage6_infer_mask_to_obb.py --config configs/stage6_infer_mask_to_obb.yaml
```
Expected:
- `eval/mask_to_obb_preds/predictions.jsonl`
- `eval/mask_to_obb_preds/previews/*.png`

## 7) Threshold calibration
```bash
python scripts/stage7_calibrate_thresholds.py --config configs/stage7_calibrate_thresholds.yaml
```
Expected:
- `runs/retfound_mask/<run_name>/calibrated_thresholds.json`

## 8) Re-run inference with calibrated thresholds
Set `postprocess.thresholds_json` in `configs/stage6_infer_mask_to_obb.yaml` to the calibrated JSON path, then rerun stage 6.

## Notes
- This pipeline uses native polygons/OBBs end-to-end for training labels and inference export.
- No COCO conversion is required in the default workflow.
