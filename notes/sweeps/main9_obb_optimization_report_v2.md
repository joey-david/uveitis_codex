# Main9 OBB Optimization Report v2

## Objective
Find any reversible change that beats:
- `out/weights/yolo_obb_main9_mmproxy_contrastive_best.pt`
- `mAP50=0.27465`, `mAP50-95=0.16176`

## What Was Tested
- YOLO11l baseline transfer (N1)
- UWF class-balanced remixes (N2)
- Val-relevant class filtering + balancing (N3)
- Seed robustness sweeps from prior best recipe (N4/N5)
- LR tweak around prior best recipe (N6a)
- Inference resolution sweep (N7)

## Best Result
No run beat the reference.

Current best remains:
- `out/weights/yolo_obb_main9_mmproxy_contrastive_best.pt`
- `mAP50=0.27465`
- `mAP50-95=0.16176`

Closest alternatives:
- N3 (`val5` filtering + balancing): `mAP50=0.23750`, `mAP50-95=0.15577`
- N4 (seed=1 repro): `mAP50=0.23719`, `mAP50-95=0.13800`

## Key Findings
- The prior best is sensitive: changing seed, LR, or balancing strategy consistently reduced mAP.
- Restricting train labels to val-present classes improved class focus but still underperformed global best.
- Inference resolution tuning did not help: 1536 remains optimal on this val split.

## Artifacts
- Checklist: `notes/sweeps/main9_obb_optimization_checklist_v2.md`
- Sweep JSON summary: `eval/sweeps/main9_sweep_results_v2.json`
- Per-run JSONs: `eval/sweeps/N*.json`
