# `uveitis_pipeline.preprocess`

Image preprocessing pipeline (ROI -> crop -> normalization -> global resize -> tiles).

## Class

| Class | Description |
|---|---|
| `SamPromptMasker` | Loads a SAM model and predicts prompted fundus masks for UWF images. |
| `Sam2PromptMasker` | Loads a SAM2 model and predicts prompted fundus masks for UWF images. |

## Functions

| Function | Description |
|---|---|
| `_compute_threshold_mask(image, cfg)` | Threshold/morphology fallback mask builder. |
| `_safe_erode(mask, erode_px)` | Erodes stats mask without collapsing to empty mask. |
| `compute_roi_mask(image, cfg, dataset='', sam_masker=None, sam2_masker=None)` | Selects ROI method (`sam2_prompted`, `sam_prompted`, or threshold), optionally per dataset. |
| `crop_to_roi(image, mask, pad_px)` | Crops image to ROI bounding box with padding and returns crop metadata. |
| `normalize_color(image, stats_mask, method, out_mask=None, ref=None)` | Applies color normalization (`zscore_rgb`, `grayworld`, `clahe_luminance`, `reinhard_lab_ref`) using stats mask and optional output mask. Reference stats are required for `reinhard_lab_ref`. |
| `resize_global(image, size)` | Pads to square, resizes to fixed global size, returns geometry metadata. |
| `tile(image, tile_size, overlap)` | Produces overlapping fixed-size tiles and per-tile global coordinates. |
| `reconstruct_from_tiles(tiles, metas, out_size)` | Reconstructs global image from tiles by averaging overlaps. |
| `process_manifest(manifest_rows, cfg, out_root)` | Runs full preprocessing pipeline for manifest rows and returns quality counters. |

## Notes
- ROI masks are cleaned with morphology and then "hole removal" by filling the largest external contour (so internal holes are eliminated).
- Photometric statistics exclude pure black pixels to avoid bias from masked-out regions and fundus borders.
