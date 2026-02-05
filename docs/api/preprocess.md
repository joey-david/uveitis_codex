# `uveitis_pipeline.preprocess`

Image preprocessing pipeline (ROI -> crop -> normalization -> global resize -> tiles).

## Functions

| Function | Description |
|---|---|
| `compute_roi_mask(image, cfg)` | Builds retina ROI mask using grayscale/saturation thresholding and morphology. |
| `crop_to_roi(image, mask, pad_px)` | Crops image to ROI bounding box with padding and returns crop metadata. |
| `normalize_color(image, roi_mask, method)` | Applies color normalization (`zscore_rgb`, `grayworld`, or `clahe_luminance`). |
| `resize_global(image, size)` | Pads to square, resizes to fixed global size, returns geometry metadata. |
| `tile(image, tile_size, overlap)` | Produces overlapping fixed-size tiles and per-tile global coordinates. |
| `reconstruct_from_tiles(tiles, metas, out_size)` | Reconstructs global image from tiles by averaging overlaps. |
| `process_manifest(manifest_rows, cfg, out_root)` | Runs full preprocessing pipeline for manifest rows and returns quality counters. |
