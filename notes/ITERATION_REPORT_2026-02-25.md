# Iteration Report (2026-02-25)

## Objective
Drive up native mask-first UWF lesion localization/classification performance with repeated train/eval/adapt loops.

## Setup Used
- Data prep: `preproc_main9_fastiter` from:
  - `manifests/fgadr_train700.jsonl`
  - `manifests/uwf700_labeled.jsonl`
- Label space: `configs/active_label_space.yaml` (main detector classes only)
- Native labels: `labels_native_main9_fastiter`
- Backbone: RETFound ViT-L (`models/retfound/RETFound_cfp_weights.pth`)
- Adaptation: `runs/retfound_adapt/main9_iterA/best.pt`

## Key Obstacles and Fixes
1. GPU in docker reported unavailable with `--gpus all` only.
- Fix: always run with `--runtime=nvidia --gpus all`.

2. Stage-5 mask training initially collapsed to near-all-zero predictions.
- Root cause: Dice over all classes (including absent classes) and sparse positives heavily penalized positive predictions.
- Fix: updated `masked_bce_dice_loss` in `src/uveitis_pipeline/retfound_mask.py`:
  - class-wise `pos_weight` BCE
  - dice emphasis on present classes
  - downweighted absent-class dice term

3. Adapted checkpoint loading failed when input-size changed (positional embedding mismatch).
- Fix: added `load_encoder_state(...)` interpolation path and robust state extraction for `adapt_ckpt` loading.

4. Calibration overpredicted absent classes on val.
- Fix: `scripts/stage7_calibrate_thresholds.py` now sets absent-class threshold to high value (`absent_class_threshold`, default `0.95`).

## Iterations
- IterB (tile-focused, 640):
  - `map50`: `0.0726`
  - `macro_f1`: `0.0302`

- IterC (global-focused, 896):
  - raw calibrated: `map50=0.1073`, `macro_f1=0.0335`
  - best postprocess (class-cap): `map50=0.1073`, `macro_f1=0.0559`

- IterD (UWF-only finetune, 1024):
  - `map50=0.0589`, `macro_f1=0.0469` (worse than IterC)

- IterE (focal variant):
  - early trend degraded; run stopped (not promoted).

## Current Best
- Model checkpoint: `runs/retfound_mask/main9_iterC/best.pt`
- Inference config: `configs/experiments/main9_best_infer.yaml`
- Best predictions + metrics:
  - `eval/main9_best_preds_v2/predictions.jsonl`
  - `eval/main9_best_preds_v2/metrics_ap50_v2.json`
  - `eval/main9_best_preds_v2/previews/`

## Best Postprocess Settings
- `thresholds_json`: `runs/retfound_mask/main9_iterC/calibrated_thresholds.json`
- `min_area_px`: `64`
- `nms_iou`: `0.20`
- `max_preds_per_class`: `2`

## Commands (Best Path)
```bash
python scripts/stage6_infer_mask_to_obb.py --config configs/experiments/main9_best_infer.yaml
python scripts/eval_native_detection.py \
  --gt-index labels_native_main9_fastiter/uwf700_val_global.jsonl \
  --pred-jsonl eval/main9_best_preds_v2/predictions.jsonl \
  --out eval/main9_best_preds_v2/metrics_ap50_v2.json --iou 0.5 --score 0.0
```

## Late Iterations (Post-native-baseline)

### 1) Detection-level calibration
- Implemented `scripts/stage7_calibrate_detection_postprocess.py` to tune per-class threshold + min-area directly on detection AP/F1 (instead of pixel IoU only).
- Wired stage6 to consume `postprocess.class_postprocess_json` overrides.
- IterC improvement:
  - before: `map50=0.1073`, `macro_f1=0.0559`
  - after detection-calib: `map50=0.1239`, `macro_f1=0.0561`
  - file: `eval/main9_iterC_detcalib_preds/metrics_ap50_v2.json`

### 2) Additional inference ablations
- Added optional stage6 TTA (`model.tta_scales`, `model.tta_hflip`) and `postprocess.component_mode=union`.
- TTA and union mode did not improve AP compared with per-class extraction on iterC.

### 3) Training ablations (iterG / iterJ)
- Added sampler and class-weight controls:
  - `NativeMaskDataset.build_balanced_sampler(mode/power/empty_weight)`
  - stage5 supports `auto_class_weights` + class-weighted loss.
- IterG (1024 + global+UWF tiles) underperformed (`map50=0.0746`), discarded.
- IterJ (short fine-tune from iterC with class-weighting) produced `map50=0.2337` but severe class collapse (`hyalite AP=0`), so discarded as standalone.

### 4) Class-wise ensemble (best AP on val)
- Brute-force class-source merge between iterC-detcalib and iterJ.
- Best choice: use iterJ only for `oedeme_papillaire`, iterC for all other classes.
- Final val metrics:
  - `map50=0.2739`
  - `weighted_ap50=0.1706`
  - `macro_f1=0.0548`
- Artifacts:
  - `eval/main9_ensemble_best_preds/predictions.jsonl`
  - `eval/main9_ensemble_best_preds/metrics_ap50_v2.json`
  - `eval/main9_ensemble_best_preds/choice.json`
  - `eval/main9_ensemble_best_preds/previews/`

### 5) Reproducibility helpers added
- `scripts/ensemble_native_predictions.py`: deterministic class-wise merge of two prediction jsonl files.
- `scripts/qa_native_pred_overlays.py`: GT+prediction preview rendering for native pipeline.
- Ensemble class map stored at `configs/experiments/main9_ensemble_choice_iterC_iterJ.json`.
