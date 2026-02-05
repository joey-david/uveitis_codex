# Uveitis Codex

Pipeline for UWF uveitis lesion localization/classification with a RetFound-adapted Faster R-CNN workflow.

## Documentation

- Markdown docs entrypoint: `docs/index.md`
- Optional rendered docs site (MkDocs):
  - `pip install -r requirements-docs.txt`
  - `mkdocs serve`

## What is implemented

- Stage 0 data pipeline:
  - unified manifest + split builder
  - retina ROI mask/crop/normalization
  - global resize + tile export + tile metadata
  - label harmonization (FGADR mask -> HBB, UWF OBB -> HBB) to COCO
- Stage 1/2/3 detector scaffold:
  - YAML-driven training with checkpoints/metrics
  - RetFound backbone adapter (from `../RETFound`) + FPN + Faster R-CNN
  - tile inference + merge-to-global + NMS
- Stage 4 optional hooks:
  - pseudo-label dataset expansion script
  - optional RetFound MAE continuation hook (runs if `../RETFound/main_pretrain.py` exists)
- Bottleneck reports:
  - dataset/preproc/training/ablation scripts

## Repository layout (new)

- `src/uveitis_pipeline/`: core pipeline modules
- `scripts/`: stage entrypoints
- `configs/`: YAML configs per stage
- `manifests/`, `splits/`: ingestion outputs
- `preproc/`: Stage 0 image artifacts
- `labels_coco/`, `labels_debug/`: detector labels + debug overlays
- `runs/`, `preds/`, `preds_vis/`, `eval/`, `pseudo_labels/`: training/inference/eval outputs

## Local quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## End-to-end command flow

### 0. Build manifests + split file

```bash
python scripts/stage0_build_manifest.py --config configs/stage0_manifest.yaml
```

Expected stdout:
- `Wrote manifests to manifests`
- `Wrote split file to splits/stage0_0.json`
- per-dataset counts (split + label types + classes)

Output files:
- `manifests/{dataset}.jsonl`
- `manifests/{dataset}.csv`
- `splits/stage0_0.json`

### 1. Run preprocessing (ROI/crop/normalize/tile)

```bash
python scripts/stage0_preprocess.py --config configs/stage0_preprocess.yaml
```

Expected stdout:
- `Preprocessing complete`
- metrics dict with ROI fail rates, tile distribution, reconstruction error

Output files:
- `preproc/roi_masks/{image_id}.png`
- `preproc/crops/{image_id}.png`
- `preproc/crop_meta/{image_id}.json`
- `preproc/norm/{image_id}.png`
- `preproc/norm_meta/{image_id}.json`
- `preproc/global_1024/{image_id}.png`
- `preproc/tiles/{image_id}/{tile_id}.png`
- `preproc/tiles_meta/{image_id}.json`
- `preproc/verify/*.png`, `preproc/verify/preprocess_metrics.json`

### 2. Build COCO labels for detector training

```bash
python scripts/stage0_build_labels.py --config configs/stage0_labels.yaml
```

Expected stdout:
- per-dataset/per-split COCO summaries (`num_images`, `num_annotations`, class counts)

Output files:
- `labels_coco/{dataset}_{split}.json`
- `labels_coco/{dataset}_{split}_tiles.json`
- `labels_coco/summary.json`
- `labels_debug/{dataset}_{split}/*.png`

### 3. Overfit smoke checkpoint (10 images)

```bash
python scripts/train_detector.py --config configs/train_overfit10.yaml
```

Output files:
- `runs/overfit_10/config.yaml`
- `runs/overfit_10/checkpoints/epoch_*.pth`
- `runs/overfit_10/checkpoints/best.pth`
- `runs/overfit_10/metrics.jsonl`
- `runs/overfit_10/val_report.json`

### 4. FGADR pretrain

```bash
python scripts/train_detector.py --config configs/train_fgadr.yaml
```

Output files:
- `runs/fgadr_pretrain/...`

### 5. Uveitis fine-tune (lax-box settings)

```bash
python scripts/train_detector.py --config configs/train_uveitis_ft.yaml
```

Output files:
- `runs/uveitis_ft/...`
- `runs/uveitis_ft/val_report.json` (includes sensitivity @ FP/image)

### 6. Tile inference + global merge

```bash
python scripts/infer_detector.py --config configs/infer_uveitis_ft.yaml
```

Output files:
- `preds/uveitis_ft/{image_id}.json`
- `preds_vis/uveitis_ft/{image_id}.png`

### 7. Reports / ablations

```bash
python scripts/report_dataset.py --manifests manifests/uwf700.jsonl manifests/fgadr.jsonl --cocos labels_coco/uwf700_train.json labels_coco/uwf700_train_tiles.json --out eval/report_dataset.json
python scripts/report_preproc.py --manifest manifests/uwf700.jsonl --preproc-root preproc --out-dir eval/preproc
python scripts/report_training.py --run-dir runs/uveitis_ft --out-json eval/training_report.json --out-png eval/training_curves.png
python scripts/ablate_preproc.py --pred-a preds/uveitis_ft --pred-b preds/uveitis_ft_no_norm --out eval/ablate_preproc.json
```

### 8. Optional scripts

```bash
python scripts/stage4_continue_mae.py --retfound-dir ../RETFound --data-path preproc/global_1024
python scripts/stage4_pseudo_label_expand.py --base-coco labels_coco/uwf700_train_tiles.json --pred-dir preds/uveitis_ft --out-coco pseudo_labels/uveitis_ft/train_plus_pseudo.json
```

## RetFound integration notes

- `src/uveitis_pipeline/modeling.py` loads RetFound code from `../RETFound`.
- If you have checkpoint weights, set `model.retfound_ckpt` in training/inference YAML.
- If no RetFound checkpoint is supplied, code still runs (backbone init path remains valid).

## Docker

Use `docker.md` for full remote A100 flow (build/run/train/evaluate) with expected outputs and artifact locations.
