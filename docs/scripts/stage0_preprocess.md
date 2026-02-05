# `stage0_preprocess.py`

## Purpose
Run Stage 0 preprocessing (ROI mask, crop, normalization, global resize, tiles) and write verification artifacts.

## CLI
```bash
python scripts/stage0_preprocess.py --config configs/stage0_preprocess.yaml [--manifest PATH] [--max-images N]
```

## Reads
- Input manifests listed in config (or `--manifest` override).

## Writes
- `preproc/roi_masks`, `preproc/crops`, `preproc/crop_meta`
- `preproc/norm`, `preproc/norm_meta`
- `preproc/global_1024`, `preproc/tiles`, `preproc/tiles_meta`
- `preproc/verify/*`

## Functions
| Function | Description |
|---|---|
| `_overlay_boundary(image, mask_rgb)` | Draws ROI boundary edges over the raw image for montage QA. |
| `_preview(image, max_side)` | Downscales an image for compact montage visualization. |
| `main()` | Loads manifests/config, runs preprocessing, writes verification montages and reconstruction metrics. |

## Core module dependencies
- [`uveitis_pipeline.preprocess`](../api/preprocess.md)
- [`uveitis_pipeline.common`](../api/common.md)
