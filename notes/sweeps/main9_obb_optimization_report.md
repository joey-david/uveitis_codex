# Main9 OBB Optimization Report (A100 Sweep)

## Scope
- Objective: beat current best Main9 UWF val performance.
- Fixed eval protocol: `out/yolo_obb/uwf700_tiles_main9/data.yaml`, split `val`, `imgsz=1536`, `batch=4`.
- Baseline reference checkpoint: `out/weights/yolo_obb_main9_mmproxy_contrastive_best.pt`.

## Critical Finding Before Sweeps
- `preproc/tiles` had been cleaned, which broke many symlinked YOLO datasets (especially FGADR-based mixes).
- I regenerated `preproc/tiles/*` from `preproc/global_1024` + `preproc/tiles_meta` so mixed datasets became valid again.

## Experiments Run
- Prior runs (E1-E6): all underperformed vs reference.
- New runs in this cycle:
  - `E7_mixpos_from_mmproxy_lr1e4_e6` (fgadr1+uwfpos30+uwfbg10 mixed): `mAP50=0.1902`, `mAP50-95=0.1201`.
  - `E8_mmproxy_medaug_lr8e5_e12` (low HSV/geo/no flips): `mAP50=0.2047`, `mAP50-95=0.1353`.
  - `E9_balmix1200_uwfmm12_lr1e4_e4` (balanced mix with FGADR sample): `mAP50=0.1778`, `mAP50-95=0.0891`.
  - `E10_mmproxy_polish_freeze10_lr2e5_e20` (very-low-LR freeze polish): `mAP50=0.2288`, `mAP50-95=0.1330`.
  - `E11` inference-time TTA check: no change (Ultralytics OBB val reports `augment=True` unsupported).

## Best Model (Unchanged)
- Best checkpoint remains:
  - `out/weights/yolo_obb_main9_mmproxy_contrastive_best.pt`
- Performance:
  - `mAP50=0.27465`
  - `mAP50-95=0.16176`

## Artifacts
- Sweep checklist: `notes/sweeps/main9_obb_optimization_checklist.md`
- Sweep machine-readable results: `eval/sweeps/main9_sweep_results.json`
- Per-run JSON summaries:
  - `eval/sweeps/E7_mixpos_from_mmproxy_lr1e4_e6.json`
  - `eval/sweeps/E8_mmproxy_medaug_lr8e5_e12.json`
  - `eval/sweeps/E9_balmix1200_uwfmm12_lr1e4_e4.json`
  - `eval/sweeps/E10_mmproxy_polish_freeze10_lr2e5_e20.json`
  - `eval/sweeps/E11_tta_eval.json`
