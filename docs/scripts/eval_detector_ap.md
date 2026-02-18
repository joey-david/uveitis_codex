# `eval_detector_ap.py`

## Purpose
Compute a real (score-swept) AP/mAP for a detector checkpoint on a COCO val set, without `pycocotools`.

## CLI
```bash
python scripts/eval_detector_ap.py \
  --ckpt runs/<run_name>/checkpoints/best.pth \
  --coco labels_coco/uwf700_val_tiles.json \
  --iou 0.5 \
  --max-dets 400
```

## Reads
- Detector checkpoint (`.pth`) produced by `train_detector.py`.
- COCO labels JSON.

## Writes
- Nothing (prints a JSON dict with `mAP` and per-class AP).

