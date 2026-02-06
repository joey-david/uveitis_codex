# Training Progress Log

This file is a running engineering log for training the UWF symptom detector.

## Goal

- Final task: localize + classify uveitis-related symptoms on UWF images.
- Input: `preproc/tiles/*/*.png` produced by stage0 preprocessing (SAM2 ROI for UWF, mask-first photometric normalization).
- Labels: COCO JSON built from UWF700 OBB annotations (+ optionally FGADR masks converted to boxes).
- Model: Faster R-CNN with RETFound ViT-L backbone adapted to a simple FPN neck, with early blocks frozen initially.

## Conventions

- Run outputs: `runs/<run_name>/` (TensorBoard in `runs/<run_name>/tb/`, metrics in `runs/<run_name>/metrics.jsonl`).
- Curves: `python scripts/report_training.py --run-dir runs/<run_name> --out-json runs/<run_name>/report.json --out-png runs/<run_name>/curves.png`

## Timeline (Fill In As We Go)

- RETFound vendored under `third_party/RETFound_MAE/` and weights downloaded to `models/retfound/RETFound_cfp_weights.pth` (stripped to model-only).
- FGADR preprocessing: `docker compose run --rm train python scripts/stage0_preprocess.py --manifest manifests/fgadr.jsonl`
- FGADR COCO labels build: `docker compose run --rm train python scripts/stage0_build_labels.py --config configs/stage0_labels.yaml`
- Smoke sanity: `configs/train_overfit10.yaml`
- FGADR pretrain: `configs/train_fgadr.yaml`
- UWF finetune: `configs/train_uveitis_ft.yaml`

### 2026-02-06

- Smoke run (`runs/uwf_vitl_smoke/`): 2 epochs on 16-image overfit subset at 512px; ran end-to-end, but `val_mAP_proxy=0.0` (likely too short / needs better score thresholds or longer training).
- FGADR pretrain run (`runs/fgadr_vitl_1h/`): 12 epochs on a 1200/300 subsample of FGADR tiles at 768px. Best `val_mAP_proxy=0.0143` at epoch 8 (many false positives at default inference threshold).
- UWF finetune run (`runs/uveitis_from_fgadr_vitl_1h/`): 30 epochs on UWF uveitis tiles (initialized from `runs/fgadr_vitl_1h/checkpoints/best.pth`), best `val_mAP_proxy=0.0215` at epoch 4.
