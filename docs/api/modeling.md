# `uveitis_pipeline.modeling`

Detector model construction, including RETFound-backed FPN adapter.

## Functions

| Function | Description |
|---|---|
| `_load_retfound_vit_l(img_size)` | Imports RETFound ViT-L backbone from `../RETFound` or `../retfound`. |
| `_clean_checkpoint_dict(state)` | Normalizes checkpoint key names and removes incompatible head keys. |
| `_interpolate_pos_embed(pos_embed, h, w)` | Interpolates positional embeddings to match current patch grid. |
| `build_detector(cfg)` | Builds Faster R-CNN model from config (RETFound or ResNet50-FPN path). |

## Class: `RetFoundSimpleFPN`

### Constructor
- `__init__(backbone, out_channels=256, ckpt_path=None, freeze_blocks=0)`

### Methods
| Method | Description |
|---|---|
| `set_freeze_blocks(n)` | Freezes first `n` transformer blocks + patch embed. |
| `forward(x)` | Produces multi-scale feature maps `p2..p5` from ViT tokens. |
