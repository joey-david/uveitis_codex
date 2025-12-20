# Switching Models (Pretrained Encoders)

Encoders are selected from `config.yaml` and propagate across stages.

## MTSN encoder (Stage 1)
- Key: `mtsn.encoder_name`
- Supported (out of the box):
  - `scratch_resnet18` (random init)
  - `resnet18_imagenet`
  - `resnet50_imagenet`
  - `efficientnet_b0_imagenet`
  - `timm:convnext_tiny` (requires `timm`)
  - `timm:vit_base_patch16_224` (requires `timm`)

Notes
- Training transforms match the encoder automatically (input size, mean/std) with light augmentations.
- Pretrained encoders are frozen for a few warmup epochs before fine-tuning.
 - Patch extraction sizes in training and graph building are aligned to the encoder input size to preserve details.

Recommended presets
- Strong default: `resnet50_imagenet`
- Modern CNN: `timm:convnext_tiny`
- ViT baseline: `timm:vit_base_patch16_224`

## Graph embeddings (Stage 3)
- Key: `gacnn.encoder_name`
- Options:
  - `use_mtsn` → use the trained MTSN encoder weights
  - any of the names above → use a fresh pretrained encoder for graph features

Changing models
1) Edit `config.yaml` fields above.
2) For Stage 1, run `python -m src.mtsn.train_mtsn`.
3) For Stage 3, rebuild graphs: `python -m src.gacnn.make_graphs`, then train `python -m src.gacnn.training`.

Fine-tuning controls (config)
- `mtsn.training.freeze_epochs`: how many warmup epochs to keep the encoder frozen (default 3). Use 0 to disable.
- `mtsn.training.encoder_lr_mult`: encoder LR multiplier when fine-tuning (e.g., 0.1 → 10× smaller LR than heads).

Tips
- Prefer `resnet50_imagenet` as a strong default on small data.
- If you add `timm` or other encoders later, extend `src/common/encoders.py::build_encoder` with a new case.
 - For specialized models (e.g., RETFound), ensure the correct normalization is used. If using a custom checkpoint not covered by torchvision/timm, add its mean/std and size in the encoder shim.
