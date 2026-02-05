# `infer_detector.py`

## Purpose
Run tile-wise detection inference and merge into global-image predictions.

## CLI
```bash
python scripts/infer_detector.py --config configs/infer_uveitis_ft.yaml
```

## Reads
- Inference YAML (`input`, `model`, `runtime`, `output`).
- Preprocessed tiles + tile metadata.
- Trained checkpoint.

## Writes
- `preds/<exp_name>/*.json`
- `preds_vis/<exp_name>/*.png` (if predictions exist)

## Functions
| Function | Description |
|---|---|
| `main()` | Loads inference config and runs inference pipeline. |

## Core module dependencies
- [`uveitis_pipeline.infer`](../api/infer.md)
- [`uveitis_pipeline.modeling`](../api/modeling.md)
