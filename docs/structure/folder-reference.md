# Folder Reference

This page documents the main folders used by the implemented pipeline.

## Source and entrypoints

| Folder | Purpose | Written by |
|---|---|---|
| `src/uveitis_pipeline/` | Core Python modules (manifest, preprocess, labels, train, infer, reports). | Manual code changes |
| `scripts/` | Stage/program entrypoints and utilities. | Manual code changes |
| `configs/` | YAML configs for each stage and smoke/profile variants. | Manual code changes |

## Dataset indexing outputs

| Folder | Purpose | Producer |
|---|---|---|
| `manifests/` | Dataset manifests in `.jsonl` and `.csv`. | `scripts/stage0_build_manifest.py` |
| `splits/` | Split files (`train`/`val`/`test` image IDs). | `scripts/stage0_build_manifest.py` |

## Preprocessing outputs

| Folder | Purpose | Producer |
|---|---|---|
| `preproc/roi_masks/` | Binary retina ROI masks as RGB PNGs. | `scripts/stage0_preprocess.py` |
| `preproc/crops/` | ROI crops from original images. | `scripts/stage0_preprocess.py` |
| `preproc/crop_meta/` | Crop metadata (`bbox_xyxy`, sizes). | `scripts/stage0_preprocess.py` |
| `preproc/norm/` | Color-normalized crop images. | `scripts/stage0_preprocess.py` |
| `preproc/norm_meta/` | Normalization stats per image (mean/std/method). | `scripts/stage0_preprocess.py` |
| `preproc/ref/` | Reference stats used by reference-based normalization (e.g. regular fundus LAB mean/std). | `scripts/build_regular_fundus_color_ref.py` |
| `preproc/global_1024/` | Square global resized images for detector space. | `scripts/stage0_preprocess.py` |
| `preproc/tiles/` | Tiled images per sample (`tile_000.png`, ...). | `scripts/stage0_preprocess.py` |
| `preproc/tiles_meta/` | Tile geometry + global mapping metadata. | `scripts/stage0_preprocess.py` |
| `preproc/verify/` | Verification plots and `preprocess_metrics.json`. | `scripts/stage0_preprocess.py` |

## Label outputs

| Folder | Purpose | Producer |
|---|---|---|
| `labels_coco/` | COCO labels for global images + tile images. | `scripts/stage0_build_labels.py` |
| `labels_debug/` | Overlay debug images for spot-checking boxes. | `scripts/stage0_build_labels.py` |

## Training, inference, evaluation outputs

| Folder | Purpose | Producer |
|---|---|---|
| `runs/` | Training run directories (`config.yaml`, checkpoints, metrics, reports). | `scripts/train_detector.py` |
| `preds/` | Inference JSON predictions per image. | `scripts/infer_detector.py` |
| `preds_vis/` | Inference overlays for predicted boxes. | `scripts/infer_detector.py` |
| `eval/` | Dataset/preproc/training/ablation reports. | `scripts/report_*.py`, `scripts/ablate_preproc.py` |
| `eval/preproc_norm_qa_*` | Preprocessing QA bundles (mask, crop, norm, tiles) on small UWF samples. | `scripts/qa_preproc_norm_to_regular.py` |
| `eval/sam2_mask_compare_*` | SAM2 masking strategy comparisons (visual strips + summaries). | `scripts/compare_sam2_masking.py` |
| `pseudo_labels/` | Pseudo-expanded COCO datasets. | `scripts/stage4_pseudo_label_expand.py` |

## Documentation

| Folder | Purpose |
|---|---|
| `docs/` | This documentation system (hierarchical Markdown with cross-links). |
