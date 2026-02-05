## Stage 0 — Data + preprocessing (deliverables first)

### 0.0 Dataset ingestion + manifest

- [x] Implemented in `src/uveitis_pipeline/manifest.py` + `scripts/stage0_build_manifest.py`; writes unified JSONL/CSV manifests and patient-grouped split JSON. Docs: [`docs/scripts/stage0_build_manifest.md`](docs/scripts/stage0_build_manifest.md), [`docs/api/manifest.md`](docs/api/manifest.md), [`docs/configs/index.md`](docs/configs/index.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* A unified dataset loader that outputs a *manifest* (CSV/JSONL) with:

  * `image_id, filepath, dataset, split, eye(L/R?)`
  * `labels_path` (optional), `label_format` (`mask|hbb|obb|grade|none`)
  * `width,height` (raw), `notes`
* Split logic (patient-level if possible) and store split assignments in the manifest.

**Output**

* `manifests/{dataset}.jsonl`
* `splits/{exp_name}_{fold}.json` (train/val/test image_ids)

**Verification**

* Script prints counts per split + per label type + per class.

---

### 0.1 Retina ROI mask + crop

- [x] Implemented in `src/uveitis_pipeline/preprocess.py` (`compute_roi_mask`, `crop_to_roi`) and run via `scripts/stage0_preprocess.py`. Docs: [`docs/scripts/stage0_preprocess.md`](docs/scripts/stage0_preprocess.md), [`docs/api/preprocess.md`](docs/api/preprocess.md), [`docs/configs/index.md`](docs/configs/index.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* `compute_roi_mask(image) -> binary_mask`:

  * downsample → threshold on intensity/saturation → largest CC → fill holes → morphological close/open
* `crop_to_roi(image, mask) -> cropped_image, crop_meta` (bbox coords, scaling)

**Output**

* `preproc/roi_masks/{image_id}.png`
* `preproc/crops/{image_id}.png`
* `preproc/crop_meta/{image_id}.json` (bbox, original size)

**Verification**

* Montage script: overlay mask boundary on raw image for 50 random samples per dataset.
* Fail-rate metric: % images where ROI area < X% or bbox touches border suspiciously.

---

### 0.2 Retina-only photometric normalization

- [x] Implemented in `src/uveitis_pipeline/preprocess.py` (`normalize_color`: `zscore_rgb|grayworld|clahe_luminance`) with ROI-only statistics and metadata export. Docs: [`docs/scripts/stage0_preprocess.md`](docs/scripts/stage0_preprocess.md), [`docs/api/preprocess.md`](docs/api/preprocess.md), [`docs/configs/index.md`](docs/configs/index.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* `normalize_color(image, roi_mask, method="zscore_rgb"|"grayworld"|"clahe_luminance")`
* Always compute stats inside ROI; outside ROI set to 0 or keep but masked later.

**Output**

* `preproc/norm/{image_id}.png` (or `.jpg`), plus
* `preproc/norm_meta/{image_id}.json` (mean/std per channel, method)

**Verification**

* Save per-dataset histograms (inside ROI) before/after normalization.
* Quick grid comparing raw vs normalized for both UWF and regular fundus.

---

### 0.3 Canonical resizing + tiling

- [x] Implemented in `src/uveitis_pipeline/preprocess.py` (`resize_global`, `tile`, `reconstruct_from_tiles`) plus coverage/reconstruction verification in `scripts/stage0_preprocess.py`. Docs: [`docs/scripts/stage0_preprocess.md`](docs/scripts/stage0_preprocess.md), [`docs/api/preprocess.md`](docs/api/preprocess.md), [`docs/configs/index.md`](docs/configs/index.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* `resize_global(image, size=1024) -> image_1024`
* `tile(image, tile_size=768, overlap=0.25) -> tiles, tile_meta`

  * tile_meta: `(tile_id, x0,y0,x1,y1, scale)` to map back to global.

**Output**

* `preproc/global_1024/{image_id}.png`
* `preproc/tiles/{image_id}/{tile_id}.png`
* `preproc/tiles_meta/{image_id}.json`

**Verification**

* Script that reconstructs the global image from tiles (sanity check alignment).
* Tile coverage report: #tiles/image distribution.

---

### 0.4 Label conversion + harmonization

- [x] Implemented in `src/uveitis_pipeline/labels.py` + `configs/class_map.yaml` + `scripts/stage0_build_labels.py`; supports FGADR masks->HBB and UWF OBB->HBB with COCO export/debug overlays. Docs: [`docs/scripts/stage0_build_labels.md`](docs/scripts/stage0_build_labels.md), [`docs/api/labels.md`](docs/api/labels.md), [`docs/configs/index.md`](docs/configs/index.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* Converters to a single detector format (COCO-style is easiest):

  * FGADR masks → connected components → HBB boxes (+ class)
  * Uveitis OBB → HBB (axis-aligned bbox enclosing OBB) + store original OBB for later
* A class mapping file: `class_map.yaml` (dataset-specific label → unified class)

**Output**

* `labels_coco/{dataset}_train.json`, `..._val.json`
* `labels_debug/{image_id}_overlay.png` (boxes drawn on normalized image)

**Verification**

* Per-dataset class counts + average box size distribution.
* Visual overlay for 50 samples/class; catch label flips immediately.

---

## Stage 1 — Baseline detector (fast “it works” checkpoint)

### 1.0 Training scaffold

- [x] Implemented in `src/uveitis_pipeline/train.py`, `src/uveitis_pipeline/modeling.py`, `scripts/train_detector.py`, and `configs/train_*.yaml`; YAML-driven Faster R-CNN with RetFound adapter, checkpoints, metrics, seeds, tensorboard/wandb hooks. Docs: [`docs/scripts/train_detector.md`](docs/scripts/train_detector.md), [`docs/api/train.md`](docs/api/train.md), [`docs/api/modeling.md`](docs/api/modeling.md), [`docs/configs/index.md`](docs/configs/index.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* Training script configurable by YAML:

  * dataset path (to normalized tiles + COCO labels)
  * backbone (`retfound_vit_b/l`)
  * detector head (Faster R-CNN default)
  * logging (tensorboard/wandb), checkpoints, reproducible seeds

**Output**

* `runs/{exp}/config.yaml`
* `runs/{exp}/checkpoints/epoch_*.pth`
* `runs/{exp}/metrics.jsonl` + tensorboard logs

**Verification**

* Overfit test: train on 10 images until near-perfect training loss + boxes align.

---

### 1.1 Inference + merge (tile → global)

- [x] Implemented in `src/uveitis_pipeline/infer.py` + `scripts/infer_detector.py`; tile prediction, global coordinate merge, class-wise NMS, JSON export and optional overlays. Docs: [`docs/scripts/infer_detector.md`](docs/scripts/infer_detector.md), [`docs/api/infer.md`](docs/api/infer.md), [`docs/configs/index.md`](docs/configs/index.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* `predict_tiles(image_id) -> tile_preds`
* `merge_tile_preds(tile_preds, tiles_meta) -> global_preds`
* NMS in global coords; export per-image predictions.

**Output**

* `preds/{exp}/{image_id}.json` (boxes, scores, classes)
* `preds_vis/{exp}/{image_id}.png` (overlay on normalized global_1024)

**Verification**

* Run on 20 images and visually inspect merges (no coordinate drift, no duplicated boxes at tile seams).

---

## Stage 2 — Supervised transfer pretraining on FGADR (lesion primitives)

### 2.0 FGADR box dataset build

- [x] Implemented through Stage 0.4 converters (`src/uveitis_pipeline/labels.py`) and FGADR mapping in `configs/class_map.yaml`; area filtering is configurable (`min_component_area`). Docs: [`docs/scripts/stage0_build_labels.md`](docs/scripts/stage0_build_labels.md), [`docs/api/labels.md`](docs/api/labels.md), [`docs/configs/index.md`](docs/configs/index.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* Use Stage 0.4 to generate FGADR COCO boxes from masks.
* Decide whether to merge small components or keep all (start simple: keep all; optionally filter tiny specks below area threshold).

**Output**

* `labels_coco/fgadr_boxes_{train,val}.json`
* FGADR overlay debug images.

**Verification**

* Spot-check each FGADR class: boxes match lesions.

---

### 2.1 Train detector on FGADR

- [x] Training path implemented with dedicated config `configs/train_fgadr.yaml` and output structure in `runs/fgadr_pretrain/...` (code ready; full long run is command-driven). Docs: [`docs/scripts/train_detector.md`](docs/scripts/train_detector.md), [`docs/api/train.md`](docs/api/train.md), [`docs/configs/index.md`](docs/configs/index.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* Train RetFound+SimpleFPN+FasterRCNN on FGADR tiles.
* Keep augmentations photometric-heavy; mild geometric only.

**Output**

* `runs/fgadr_pretrain/...` checkpoints + metrics
* `eval/fgadr_pretrain/mAP_report.json`

**Verification**

* Qualitative: overlays show correct lesion localization.
* Quantitative: mAP on FGADR val (don’t chase SOTA; chase “works”).

---

## Stage 3 — Target fine-tune on UWF uveitis (the actual goal)

### 3.0 Uveitis dataset build (HBB target)

- [x] Implemented via UWF OBB->HBB conversion in `src/uveitis_pipeline/labels.py` and exported as COCO through `scripts/stage0_build_labels.py`. Docs: [`docs/scripts/stage0_build_labels.md`](docs/scripts/stage0_build_labels.md), [`docs/api/labels.md`](docs/api/labels.md), [`docs/configs/index.md`](docs/configs/index.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* Convert your uveitis OBBs → HBB, using the same COCO format.
* Optional: store a “soft box mask” target generator for later (but not required for v1).

**Output**

* `labels_coco/uveitis_{train,val}.json`
* `labels_debug/uveitis_overlay/...`

**Verification**

* Ophthalmologist sanity check on 10 images if possible.

---

### 3.1 Fine-tuning with lax-box-friendly settings

- [x] Implemented in config/code path: `configs/train_uveitis_ft.yaml` (lower IoU thresholds, higher proposal counts, class-balanced sampling, freeze/unfreeze schedule, lower LR) + `src/uveitis_pipeline/train.py`. Docs: [`docs/scripts/train_detector.md`](docs/scripts/train_detector.md), [`docs/api/train.md`](docs/api/train.md), [`docs/configs/index.md`](docs/configs/index.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* Initialize from FGADR-pretrained checkpoint.
* Modify training config:

  * Lower IoU thresholds (RPN/ROI positive) e.g. 0.4–0.5
  * Keep more proposals (higher recall)
  * Class-balanced sampling or loss reweighting
  * Freeze early ViT blocks for first N epochs, then unfreeze
  * Lower LR and shorter schedule

**Output**

* `runs/uveitis_ft/...` checkpoints + metrics
* `eval/uveitis_ft/val_report.json` + per-class sensitivity at FP/image

**Verification**

* Qualitative overlays on val: do boxes land “in the right neighborhoods”?
* Quantitative: per-class recall at fixed FP/image (more meaningful than strict IoU).

---

## Stage 4 — Optional upgrades (only after baseline is solid)

### 4.A Optional: Self-supervised UWF adaptation (only if easy)

- [x] Hook implemented in `scripts/stage4_continue_mae.py`; auto-checks for `../RETFound/main_pretrain.py` and runs continuation command when available. Docs: [`docs/scripts/stage4_continue_mae.md`](docs/scripts/stage4_continue_mae.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* Continue MAE pretraining of RetFound on unlabeled UWF normalized crops/tiles.

**Output**

* `pretrain_runs/mae_uwf/...` adapted backbone weights

**Verification**

* Linear probe / quick downstream improvement check on a small supervised task, or just compare downstream fine-tune curves.

---

### 4.B Optional: pseudo-label expansion

- [x] Implemented in `scripts/stage4_pseudo_label_expand.py`; merges high-confidence predictions into COCO pseudo-label training set. Docs: [`docs/scripts/stage4_pseudo_label_expand.md`](docs/scripts/stage4_pseudo_label_expand.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).

**Implement**

* Run best model on unlabeled UWF, keep high-confidence predictions, retrain briefly.

**Output**

* `pseudo_labels/{exp}/...json`
* `runs/{exp}_pl/...`

**Verification**

* Ensure pseudo-labels aren’t exploding in count or drifting to artifacts (lashes).

---

### 4.C Optional: bring back OBB (if one class truly directional)

- [ ] Not implemented yet; no rotated-box head added in this pass.

**Implement**

* Add rotated box head only for that class; keep rest HBB.

**Output**

* Rotated predictions + overlays.

**Verification**

* Compare against HBB baseline on that class only.

---

## Bottleneck identification hooks (build these in early)

- [x] Implemented. Docs: [`docs/scripts/report_dataset.md`](docs/scripts/report_dataset.md), [`docs/scripts/report_preproc.md`](docs/scripts/report_preproc.md), [`docs/scripts/report_training.md`](docs/scripts/report_training.md), [`docs/scripts/ablate_preproc.md`](docs/scripts/ablate_preproc.md), [`docs/api/reports.md`](docs/api/reports.md), [`docs/structure/stage-map.md`](docs/structure/stage-map.md).
  - `scripts/report_dataset.py` (`src/uveitis_pipeline/reports.py::report_dataset`)
  - `scripts/report_preproc.py` (`report_preproc`)
  - `scripts/report_training.py` (`report_training`)
  - `scripts/ablate_preproc.py` (`ablate_preproc`)

* `report_dataset.py`: counts, class imbalance, average lesion size, ROI failure rate
* `report_preproc.py`: before/after color stats per dataset, sample grids
* `report_training.py`: loss curves, per-class AP/recall, FP/image
* `ablate_preproc.py`: run inference with/without normalization and ROI masking on a fixed mini-set and compare outputs

If you want, I can also give you a suggested folder structure + exact JSON schemas (manifest, crop_meta, tiles_meta, COCO annotations) so every stage stays plug-and-play.
