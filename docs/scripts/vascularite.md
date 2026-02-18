# Vascularite Tooling (`scripts/vascularite/*`)

## Purpose
`vascularite` is handled outside of the main detector due to sparse labels and different geometry.

This repo provides a small, fast vesselness-based module (no training required) and a few training/inference utilities under `scripts/vascularite/`.

## Key policy
- Main detector label space excludes `vascularite`.
- `vascularite` is predicted by a dedicated module/model.
- Canonical policy files:
  - `configs/active_label_space.yaml`
  - `configs/main_detector_classes.txt`

## Module
- `src/uveitis_pipeline/vascularite.py`
  - `nonblack_mask`: fundus pixels (not padded black).
  - `vessel_mask`: training-free vesselness proxy (green-channel CLAHE + multi-scale blackhat).
  - `mask_to_obbs`: convert a binary mask into oriented boxes via `minAreaRect`.

## Scripts
- `scripts/vascularite/train_vascularite_patch.py`: patch-level model training (if enabled by your run).
- `scripts/vascularite/infer_vascularite_patch.py`: patch-level inference and visualization.
- `scripts/vascularite/eval_vascularite_obb_ap.py`: OBB AP evaluation for vascularite-only predictions.

## Outputs
These scripts typically write under `runs/`, `preds/`, `preds_vis/`, or `eval/` depending on the entrypoint.

