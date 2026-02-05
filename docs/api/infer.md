# `uveitis_pipeline.infer`

Inference over tiles with global merge.

## Functions

| Function | Description |
|---|---|
| `_load_model(ckpt_path, cfg, device)` | Instantiates detector and loads checkpoint weights. |
| `predict_tiles(model, image_id, preproc_root, device, score_thresh)` | Runs model on all tiles of one image and projects boxes to global coordinates. |
| `merge_tile_preds(tile_preds, iou_thresh)` | Per-class NMS merge in global coordinates. |
| `run_inference_from_config(cfg)` | Config-driven inference loop, JSON prediction output, optional visualization output. |
