# Main9 OBB Optimization Checklist v2 (Anything/Everything Sweep)

## Goal
Beat current best on UWF Main9 val:
- `out/weights/yolo_obb_main9_mmproxy_contrastive_best.pt`
- `mAP50=0.27465`, `mAP50-95=0.16176`

## Fixed Eval Protocol
- Val data: `out/yolo_obb/uwf700_tiles_main9/data.yaml`
- Eval image size: `1536`
- Eval batch: `4`
- Device: `0` (A100)

## New Checklist
- [x] N1: YOLO11l-OBB baseline on `uwf700_tiles_main9_mmproxy`
- [x] N2: UWF-only class-balance mix on `uwf700_tiles_main9_mmproxy_balanced`
- [x] N3: Train on val-relevant classes only (`hyalite/hemorragie/nodule/exudats/oedeme`) + balance
- [x] N4: Reproducibility seed stress-test from the previous best recipe
- [x] N5: Additional seed sweep (`2/3/4`) with 2-epoch quick scan
- [x] N6: LR tweak around prior best recipe (`lr0=1.5e-4`)
- [x] N7: Inference-size sweep (`1024/1280/1536/1792/2048`)

## Result Table
| Exp | Key settings | Best epoch | mAP50 | mAP50-95 | Status |
|---|---|---:|---:|---:|---|
| Ref-mmproxy | `y8l 1536 AdamW + contrastive proxy` | 2 | 0.27465 | 0.16176 | reference |
| N1 | `y11l, mmproxy, e12 (stopped)` | 4 | 0.17572 | 0.11340 | no gain |
| N2 | `y8l <- mmproxy_best, balanced mix, e4 (stopped)` | 2 | 0.16107 | 0.10219 | no gain |
| N3 | `y8l <- mmproxy_best, val5-only + balanced, e4 (stopped)` | 1 | 0.23750 | 0.15577 | no gain |
| N4 | `seed=1 repro from yolo_obb_main9_best, e5 (stopped)` | 2 | 0.23719 | 0.13800 | no gain |
| N5-seed2 | `seed=2, repro e2` | 2 | 0.20964 | 0.13187 | no gain |
| N5-seed3 | `seed=3, repro e2` | 2 | 0.20209 | 0.12316 | no gain |
| N5-seed4 | `seed=4, repro e2` | 2 | 0.20169 | 0.12655 | no gain |
| N6a | `lr0=1.5e-4, repro e2` | 2 | 0.18892 | 0.12146 | no gain |
| N7 | `inference imgsz sweep` | 1536 | 0.27465 | 0.16176 | no gain |

## Notes
- Keep all experiments reversible (new run dirs + separate JSON outputs).
- Do not overwrite `out/weights/yolo_obb_main9_mmproxy_contrastive_best.pt`.
- Machine-readable sweep summary: `eval/sweeps/main9_sweep_results_v2.json`.
- Per-run JSONs: `eval/sweeps/N*.json`.
