# `uveitis_pipeline.train`

Training loop and validation metrics.

## Functions

| Function | Description |
|---|---|
| `_match_detections(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_thresh)` | Class-wise greedy TP/FP/FN matching with IoU thresholding. |
| `_evaluate(model, loader, device, iou_thresh, fp_targets)` | Runs validation and computes per-class metrics, mAP proxy, sensitivity@FP/image. |
| `train_from_config(cfg)` | End-to-end training from YAML config, checkpointing, logging, and final report export. |

## Output contract
- Writes checkpoints every epoch + `best.pth`.
- Appends epoch rows to `metrics.jsonl`.
- Writes best validation summary to `val_report.json`.
