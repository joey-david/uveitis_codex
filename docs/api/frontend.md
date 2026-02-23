# `uveitis_pipeline.frontend`

## Purpose
Single-image inference service used by frontend UI/API.

## Main class
| Class | Description |
|---|---|
| `FrontendInferenceService` | Loads model + ROI maskers and runs ROI -> normalization -> detection -> visualization pipeline. |

## Key methods
| Method | Description |
|---|---|
| `from_yaml(config_path)` | Builds service from `configs/frontend.yaml`. |
| `infer_bytes(raw, image_name, score_thresh, dataset)` | Runs full pipeline from uploaded bytes. |
| `infer_image(image, image_name, score_thresh, dataset)` | Runs full pipeline from RGB numpy image. |

## Output payload
- `predictions`: list of `{label_id, label_name, score, box_xyxy}`
- `images`: base64 PNGs for `input`, `roi_mask`, `roi_overlay`, `masked_raw`, `normalized_crop`, `detector_input`, `final_overlay`, `juxtaposed`
- `timings_sec`: stage timings (`stage1_mask_norm`, `stage2_detect`, `total`)
- `artifact_dir`: persisted run directory if enabled

## Dependencies
- [`preprocess.py`](preprocess.md)
- [`modeling.py`](modeling.md)
- [`infer.py`](infer.md)
- [`common.py`](common.md)
