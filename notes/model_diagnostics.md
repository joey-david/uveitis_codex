# Model Diagnostics (Why mAP Is Low)

This note records the checks we ran and what they imply for next steps.

## 1) Our In-Training "mAP_proxy" Was Misleading

`src/uveitis_pipeline/train.py` reported `val_mAP_proxy`, but it's **not** COCO AP/mAP:

- It is computed at a single `training.eval_score_thresh`.
- It matches boxes greedily and then uses `precision * recall` per class as an "AP-like" scalar.
- It is very sensitive to false positives at that chosen threshold.

To get a real, score-swept AP, we added `scripts/eval_detector_ap.py`.

## 2) Box Size Diagnostics: FGADR Is a Small-Object Problem

We computed bbox size distributions on 768x768 tiles.

FGADR (`labels_coco/fgadr_train_tiles.json`):

- `min_side < 8px`: ~39.6% of boxes
- `min_side < 16px`: ~76.8% of boxes
- `min_side < 32px`: ~91.4% of boxes

This makes `mAP@0.5` intrinsically hard: a 1-2px localization error can drop IoU below 0.5 on tiny boxes.

## 3) Label QA: Boxes Look Correct

We rendered random GT overlays (so we can rule out totally broken label projection/tiling):

- `eval/diagnostics/gt_overlays_fgadr_tiles/`
- `eval/diagnostics/gt_overlays_uwf_tiles/`

## 4) Real AP Comparisons (Subset Eval)

All AP numbers below are computed with `scripts/eval_detector_ap.py` on a subset of 200 FGADR val tiles.

### FGADR: ResNet50-FPN FasterRCNN vs RETFound(ViT-L) FasterRCNN

- RETFound(ViT-L) run `runs/fgadr_vitl_1h/`:
  - `mAP@0.5`: ~0.041
  - `mAP@0.3`: ~0.087

- ResNet50-FPN run `runs/fgadr_r50_smoke/` (only ~400 train tiles, short run):
  - `mAP@0.5`: ~0.095
  - `mAP@0.3`: ~0.154

Conclusion: **our current RETFound+ViT detector stack is materially worse for FGADR**, and the gap shows up on a real AP computation.

## 5) Why RETFound(ViT-L) Underperforms on FGADR

The dominant reason is resolution / stride:

- The RETFound ViT uses `patch16`, i.e. a native **stride 16** token grid.
- Our "p2" is just an upsample of that stride-16 feature map, so it does not add new spatial detail.
- FGADR lesions are often only 3-10 pixels wide on tiles, which is below what stride-16 features can localize reliably.

This also explains why anchor tweaks didn't help: the feature map doesn't contain enough high-frequency signal for tiny boxes.

## 6) RetinaNet Attempt

We tried a RetinaNet smoke run (`runs/fgadr_retinanet_r50_smoke/`), but it produced near-zero AP as configured.
It also showed that its scores start very low (around ~0.03 early), so any evaluation with `eval_score_thresh=0.1` reports zero detections.

## 7) UWF700 Reality Check

UWF700 is very small (train split is 67 images / 268 tiles) spread across many classes.
This alone caps how high mAP can go without additional data, class merging, or stronger priors.

One measured checkpoint:

- RETFound finetune `runs/uveitis_from_fgadr_vitl_1h/` on full UWF700 val tiles:
  - `mAP@0.5`: ~0.125
  - `mAP@0.3`: ~0.140

## Recommendations

1. **For FGADR-like tiny lesions**:
   - Use a backbone with stride-4/8 features (ResNet-FPN works), OR
   - Increase input resolution (so lesions are larger in pixels), OR
   - Move to segmentation-style training instead of boxes.

2. **For UWF700 / oriented labels**:
   - If oriented boxes matter for final success, consider adding an OBB detector (current pipeline trains on AABBs only).
   - Keep using RETFound for UWF if the goal is global context + larger structures, but expect data-limited mAP.

3. **Metrics**:
   - Use `scripts/eval_detector_ap.py` (or pycocotools later) for model selection, not `mAP_proxy`.

