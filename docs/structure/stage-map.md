# Stage Map

This map links stage objectives, entry scripts, configs, inputs, and outputs.

## Stage 0.0: Build manifests + splits

- Script: [`scripts/stage0_build_manifest.py`](../scripts/stage0_build_manifest.md)
- Config: [`configs/stage0_manifest.yaml`](../configs/index.md#stage0-manifest)
- Input: dataset roots in config
- Output:
  - `manifests/*.jsonl`, `manifests/*.csv`
  - `splits/stage0_*.json`

## Stage 0.1-0.3: Preprocess ROI/crop/normalize/tile

- Script: [`scripts/stage0_preprocess.py`](../scripts/stage0_preprocess.md)
- Config: [`configs/stage0_preprocess.yaml`](../configs/index.md#stage0-preprocess)
- Input: manifests from Stage 0.0
- Output:
  - `preproc/roi_masks`, `preproc/crops`, `preproc/norm`, `preproc/global_1024`, `preproc/tiles`
  - `preproc/crop_meta`, `preproc/norm_meta`, `preproc/tiles_meta`
  - `preproc/verify/preprocess_metrics.json` and montages

## Stage 0.4: Convert labels to COCO

- Script: [`scripts/stage0_build_labels.py`](../scripts/stage0_build_labels.md)
- Configs:
  - [`configs/stage0_labels.yaml`](../configs/index.md#stage0-labels)
  - smoke/local variants
- Input:
  - manifests + split file
  - class mapping YAML
  - preprocessing metadata from Stage 0.1-0.3
- Output:
  - `labels_coco/*.json`
  - `labels_debug/*/*.png`

## Stage 1-3: Train and infer detector

- Train script: [`scripts/train_detector.py`](../scripts/train_detector.md)
- Inference script: [`scripts/infer_detector.py`](../scripts/infer_detector.md)
- Train configs:
  - overfit: [`configs/train_overfit10.yaml`](../configs/index.md#training)
  - FGADR pretrain: [`configs/train_fgadr.yaml`](../configs/index.md#training)
  - uveitis fine-tune: [`configs/train_uveitis_ft.yaml`](../configs/index.md#training)
  - CPU smoke/local smoke variants
- Inference configs:
  - [`configs/infer_uveitis_ft.yaml`](../configs/index.md#inference)
  - smoke/local smoke variants
- Output:
  - `runs/<exp_name>/...`
  - `preds/<exp_name>/*.json`
  - `preds_vis/<exp_name>/*.png`

## Stage 4 (optional hooks)

- MAE continuation hook: [`scripts/stage4_continue_mae.py`](../scripts/stage4_continue_mae.md)
- Pseudo-label expansion: [`scripts/stage4_pseudo_label_expand.py`](../scripts/stage4_pseudo_label_expand.md)
- Output:
  - optional external pretrain outputs
  - `pseudo_labels/<exp>/train_plus_pseudo.json`

## Reports and bottleneck checks

- Scripts:
  - [`scripts/report_dataset.py`](../scripts/report_dataset.md)
  - [`scripts/report_preproc.py`](../scripts/report_preproc.md)
  - [`scripts/report_training.py`](../scripts/report_training.md)
  - [`scripts/ablate_preproc.py`](../scripts/ablate_preproc.md)
- Output: `eval/*.json`, `eval/**/*.png`
