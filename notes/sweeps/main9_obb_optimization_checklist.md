# Main9 OBB Optimization Checklist (Systematic Sweep)

## Goal
Improve Main9 detector performance on `out/yolo_obb/uwf700_tiles_main9/data.yaml` val split beyond current best.

## Fixed Evaluation Protocol
- Evaluator: Ultralytics `model.val`
- Val data: `out/yolo_obb/uwf700_tiles_main9/data.yaml`
- Split: `val`
- Image size: `1536`
- Batch: `4` (or lower if model requires)
- Device: `0`
- Metrics tracked: `mAP50`, `mAP50-95`

## Current Baseline
- Weights: `out/weights/yolo_obb_main9_best.pt`
- mAP50: `0.22015`
- mAP50-95: `0.13476`

## Current Improved Reference
- Weights: `out/weights/yolo_obb_main9_mmproxy_contrastive_best.pt`
- mAP50: `0.27465`
- mAP50-95: `0.16176`

## Checklist
- [x] E1: Continue from mmproxy best, lower LR, longer schedule
- [x] E2: Continue from mmproxy best, higher image size (1792)
- [x] E3: Same as E1 with `mosaic=0` (geometry-preserving fine-tune)
- [x] E4: YOLOv8x-obb transfer on mmproxy data
- [x] E5: YOLOv8l on mmproxy with smaller LR + cosine restart (short run)
- [x] E6: mmproxy dataset with stronger proxy transform (higher CLAHE)
- [x] E7: mmproxy + positive-focused mixing (FGADR+UWF mixed)
- [x] E8: Medical-safe aug tuning on mmproxy (lower color/geo aug)
- [x] E9: Balanced mixed dataset (downsample FGADR + strong UWF ratio)
- [x] E10: Best-of-sweep long polish run
- [x] E11: Inference-side TTA check on best checkpoint(s)

## Result Table
| Exp | Key settings | Best epoch | mAP50 | mAP50-95 | Status |
|---|---|---:|---:|---:|---|
| Ref-mmproxy | `y8l 1536 AdamW lr2e-4 cos` | 2 | 0.27465 | 0.16176 | done |

| E1 | `continue mmproxy, lr1e-4, 1536, e12` | 3 | 0.22106 | 0.13437 | no gain |
| E2 | `continue mmproxy, 1792, batch2, e8` | 1 | 0.20492 | 0.14020 | no gain |
| E3 | `continue mmproxy, mosaic0, 1536, e8` | 2 | 0.20627 | 0.12842 | no gain |
| E4 | `y8x-obb from pretrained, 1536, batch2, e8` | 2 | 0.11370 | 0.06513 | no gain |
| E5 | `continue mmproxy, lr3e-4, 1536, e6` | 2 | 0.22575 | 0.12594 | no gain |
| E6 | `mmproxy strong clip4.0, 1536, e6` | 2 | 0.22320 | 0.13269 | no gain |
| E7 | `mixed fgadr1 + uwfpos30 + uwfbg10, cont. lr1e-4` | 1 | 0.19021 | 0.12010 | no gain |
| E8 | `mmproxy med-aug (low hsv/geo, no flip), lr8e-5` | 2 | 0.20465 | 0.13525 | no gain |
| E9 | `fgadr sample1200 + uwfmmproxyx12, cont. lr1e-4` | 1 | 0.17779 | 0.08911 | no gain |
| E10 | `mmproxy polish freeze10 lr2e-5 (2 epochs observed)` | 2 | 0.22881 | 0.13299 | no gain |
| E11 | `val augment=True (TTA)` | n/a | 0.27465 | 0.16176 | no change |

## Notes
- `preproc/tiles` had been cleaned; FGADR-linked YOLO datasets became fully broken until tiles were regenerated from `preproc/global_1024` + `preproc/tiles_meta`.
- Tile regeneration restored all symlinked YOLO datasets under `out/yolo_obb/*` to a usable state.
- Ultralytics OBB path currently warns `augment=True` is unsupported in `val`, so TTA does not change metrics.
