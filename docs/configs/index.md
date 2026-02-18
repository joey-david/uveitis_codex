# Config Catalog

This section documents each YAML config in `configs/`.

## Stage0 manifest

Files:
- `configs/stage0_manifest.yaml`
- `configs/stage0_manifest_smoke.yaml`
- `configs/stage0_manifest_fast.yaml` (same datasets as full, but `read_image_size: false` for faster scanning)

Consumed by:
- [`scripts/stage0_build_manifest.py`](../scripts/stage0_build_manifest.md)

Key schema:
- `seed`: int, split seed base.
- `fold`: int, fold index added to seed.
- `read_image_size`: bool, optional, skip expensive size reads when false.
- `split_ratios.train|val|test`: float, random patient-level split ratios.
- `datasets.<name>.enabled`: bool.
- `datasets.<name>.root`: path.
- `output.manifest_dir`: path.
- `output.split_dir`: path.
- `output.exp_name`: string used in split file naming.

## Stage0 preprocess

File:
- `configs/stage0_preprocess.yaml`

Consumed by:
- [`scripts/stage0_preprocess.py`](../scripts/stage0_preprocess.md)

Key schema:
- `input.manifests`: list[path], manifest JSONL files.
- `output.preproc_root`: path.
- `roi.method`: default ROI method (e.g. `threshold`).
- `roi.method_by_dataset.<dataset>`: optional per-dataset override (e.g. `uwf700: sam_prompted`).
- `roi.downsample_max_side`: int.
- `roi.sat_threshold`: int.
- `roi.crop_pad_px`: int.
- `roi.sam.checkpoint`: SAM checkpoint path.
- `roi.sam.model_type`: SAM backbone key (`vit_h`, `vit_l`, `vit_b`).
- `roi.sam.device`: inference device (`cuda` or `cpu`).
- `roi.sam.fallback_to_threshold`: bool; use threshold mask if SAM is unavailable.
- `roi.sam.multimask_output`: bool.
- `roi.sam.points_norm`: list of normalized prompt triplets `[x_norm, y_norm, label]`.
- `roi.sam.open_kernel|close_kernel`: post-mask morphology kernel sizes.
- `roi.sam.min_area_ratio|max_area_ratio|max_border_touch_ratio`: mask quality constraints for selecting SAM output.
- `roi.sam2.checkpoint`: SAM2 checkpoint path.
- `roi.sam2.model_cfg`: SAM2 model config path (from the SAM2 package).
- `roi.sam2.device`: inference device (`cuda` or `cpu`).
- `roi.sam2.fallback_to_threshold`: bool; use threshold mask if SAM2 is unavailable.
- `roi.sam2.multimask_output`: bool.
- `roi.sam2.points_norm`: list of normalized prompt triplets `[x_norm, y_norm, label]`.
- `roi.sam2.open_kernel|close_kernel`: post-mask morphology kernel sizes.
- `roi.sam2.min_area_ratio|max_area_ratio|max_border_touch_ratio`: mask quality constraints for selecting SAM2 output.
- `normalize.method`: one of `zscore_rgb`, `grayworld`, `clahe_luminance`, `reinhard_lab_ref`.
- `normalize.stats_erode_px`: int erosion radius for stats mask (reduces border contamination in normalization stats).
- `normalize.ref.stats_path`: required when using `reinhard_lab_ref`; JSON with `lab_mean` and `lab_std` computed from regular fundus images.
- `resize.global_size`: int.
- `tiling.tile_size`: int.
- `tiling.overlap`: float in `[0,1)`.
- `verify.min_roi_area_ratio`: float.
- `verify.montage_n`: int.
- `verify.reconstruct_n`: int.

## Class mapping

File:
- `configs/class_map.yaml`

Consumed by:
- [`scripts/stage0_build_labels.py`](../scripts/stage0_build_labels.md)

Key schema:
- `categories`: ordered list of category names (COCO IDs follow this order + 1).
- `maps.uwf700.<int_class_id>`: category name mapping for OBB labels.
- `maps.fgadr.<mask_folder_name>`: category name mapping for FGADR mask folders.

## Active label space (policy)

Files:
- `configs/active_label_space.yaml` (human-readable policy and rationale)
- `configs/main_detector_classes.txt` (one class name per line; used by YOLO export tooling)

Notes:
- The main detector excludes `vascularite` (separate module/model).
- This is the canonical "do not drift" list used for Main9 experiments.

## Stage0 label building

Files:
- `configs/stage0_labels.yaml`
- `configs/stage0_labels_smoke.yaml`
- `configs/stage0_labels_local_smoke.yaml`

Consumed by:
- [`scripts/stage0_build_labels.py`](../scripts/stage0_build_labels.md)

Key schema:
- `input.manifests`: list[path].
- `input.split_json`: path.
- `input.class_map_yaml`: path.
- `input.preproc_root`: path.
- `input.target_datasets`: list[`uwf700`, `fgadr`, ...].
- `input.splits`: list[`train`, `val`, `test`].
- `build.min_component_area`: int, connected-component area threshold for masks.
- `build.min_tile_box_ratio`: float, min intersection ratio for tile annotations.
- `output.labels_dir`: path.
- `output.debug_dir`: path.

## Training

Files:
- `configs/train_overfit10.yaml`
- `configs/train_fgadr.yaml`
- `configs/train_uveitis_ft.yaml`
- `configs/train_smoke.yaml`
- `configs/train_local_smoke.yaml`

Consumed by:
- [`scripts/train_detector.py`](../scripts/train_detector.md)

Key schema:
- `seed`: int.
- `run.output_dir`: path.
- `run.name`: string run ID.
- `run.tensorboard`: bool.
- `run.wandb`: bool (optional).
- `run.wandb_project`: string (optional).
- `data.train_coco`: path.
- `data.val_coco`: path.
- `data.resize`: int or null.
- `model.backbone`: `retfound_vit_l`, `retfound_vit_b`, or `resnet50_fpn`.
- `model.retfound_ckpt`: checkpoint path or empty.
- `model.input_size`: int (RETFound backbones).
- `model.fpn_out_channels`: int.
- `model.freeze_blocks`: int.
- `model.num_classes`: int (includes background + lesion categories for detector head).
- `model.*iou*`, `model.*nms*`, `model.box_detections_per_img`: detector thresholds/caps.
- `training.device`: `cuda` or `cpu`.
- `training.init_checkpoint`: optional path.
- `training.epochs`: int.
- `training.batch_size`: int.
- `training.eval_batch_size`: int.
- `training.num_workers`: int.
- `training.lr`: float.
- `training.min_lr`: float.
- `training.weight_decay`: float.
- `training.freeze_epochs`: int.
- `training.overfit_num_images`: int.
- `training.class_balanced_sampling`: bool.
- `training.eval_iou`: float.
- `training.fp_image_targets`: list[float].

## Inference

Files:
- `configs/infer_uveitis_ft.yaml`
- `configs/infer_smoke.yaml`
- `configs/infer_local_smoke.yaml`

Consumed by:
- [`scripts/infer_detector.py`](../scripts/infer_detector.md)

Key schema:
- `input.preproc_root`: path.
- `input.image_ids_json`: path to list or split dict JSON.
- `input.split_name`: string when `image_ids_json` is a dict.
- `output.pred_dir`: path.
- `output.vis_dir`: path.
- `output.exp_name`: run key appended under output dirs.
- `model.checkpoint`: path to trained checkpoint.
- `model.backbone`, `model.num_classes`, `model.retfound_ckpt`, `model.input_size`, `model.fpn_out_channels`, `model.freeze_blocks`: model creation fields.
- `model.class_names`: ordered names for visualization labels.
- `runtime.device`: `cuda` or `cpu`.
- `runtime.score_thresh`: float.
- `runtime.nms_iou`: float.
