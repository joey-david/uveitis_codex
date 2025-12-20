# Running the Pipeline

This repo has three stages:
- Stage 1: MTSN patch encoder training (`src/mtsn/train_mtsn.py`)
- Stage 2: OVV pseudo-labeling (`src/ovv/runner.py` uses `OVVLabeler`)
- Stage 3: GACNN graph build + training (`src/gacnn/make_graphs.py`, `src/gacnn/training.py`)

All paths and hyperparameters are in `config.yaml`.

## 1) Train MTSN
- Pick an encoder in `config.yaml` → `mtsn.encoder_name`.
- Run: `python -m src.mtsn.train_mtsn`
- Output: `cfg.mtsn.paths.model_save_path`

Notes
- Training now uses encoder-native transforms automatically (size, mean/std) with light augmentations for transfer learning.
- If using ImageNet weights (e.g., `resnet50_imagenet`), the encoder is frozen for a short warmup, then unfrozen.
  - Warmup and encoder LR are configurable via `mtsn.training.freeze_epochs` and `mtsn.training.encoder_lr_mult`.

Resolution and normalization
- Default input is now 224×224 (or the encoder’s native size). Config fallbacks updated to 224 and ImageNet mean/std.
- For models with different recipes (e.g., ViT variants, RETFound), the encoder shim derives the correct mean/std from the weights when possible.

## 2) Generate pseudo-labels with OVV
- OVV uses the same encoder as MTSN and its native eval transforms.
- Run: `python -m src.ovv.runner`
- Output: YOLO-style OBB labels to `cfg.ovv.paths.output_dir`

## 3) Build graphs and train GACNN
- Graphs: `python -m src.gacnn.make_graphs`
  - Uses either the trained MTSN (`gacnn.encoder_name: use_mtsn`) or a chosen encoder name (e.g., `resnet50_imagenet`).
- Train: `python -m src.gacnn.training`
  - Auto-detects node feature dimension from graph files.

Troubleshooting
- If graphs folder is missing, run `make_graphs.py` before GACNN training.
- Check `metrics/` for CSVs and JSON summaries from each stage.

Dependencies
- Pretrained TIMM models (optional): install `timm` (already listed in `src/requirements.txt`).
