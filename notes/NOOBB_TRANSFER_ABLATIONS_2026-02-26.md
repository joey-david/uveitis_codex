# No-OBB Transfer Ablations (2026-02-26)

## Goal
Test the hypothesis: train without UWF OBB supervision (FGADR mask-only training), then transfer to representation-adjusted UWF images.

## Setup
- Training supervision: **FGADR masks only** (`labels_native_main9_fastiter/fgadr_train80_global.jsonl` train, `labels_native_main9_fastiter/fgadr_val20_global.jsonl` val).
- No UWF OBB used in train/val for these runs.
- UWF inference target: `labels_native_main9_fastiter/uwf700_all_global.jsonl` (98 labeled images, used only for post-hoc evaluation).
- Shared-class subset for transfer check: `hemorragie, exudats, macroanevrisme_arteriel, ischemie_retine` (`configs/experiments/classes_shared4.txt`).

## Runs

### A) FGADR-only base (no adaptation)
- Train config: `configs/experiments/noobb_fgadr_base_mask.yaml`
- Inference output: `eval/noobb_fgadr_base_uwf_preds/predictions.jsonl`
- Metrics:
  - all classes: `eval/noobb_fgadr_base_uwf_preds/metrics_all9.json` -> `map50=0.0`, `macro_f1=0.0`
  - shared4: `eval/noobb_fgadr_base_uwf_preds/metrics_shared4.json` -> `map50=0.0`, `macro_f1=0.0`

### B) FGADR-only + RETFound adaptation (FGADR+UWF unlabeled contrastive)
- Train config: `configs/experiments/noobb_fgadr_adapt_mask.yaml`
- Inference output: `eval/noobb_fgadr_adapt_uwf_preds/predictions.jsonl`
- Metrics:
  - all classes: `map50=1.6668e-05`, `macro_f1=9.1449e-04`
  - shared4: `map50=3.7504e-05`, `macro_f1=0.00206`

### C) B + TTA inference ablation
- Inference config: `configs/experiments/noobb_fgadr_adapt_infer_uwf_tta.yaml`
- Inference output: `eval/noobb_fgadr_adapt_uwf_tta_preds/predictions.jsonl`
- Metrics:
  - all classes: `map50=5.5006e-04`, `macro_f1=0.00119`
  - shared4: `map50=0.00124`, `macro_f1=0.00267`
- Non-zero AP class in this run: `exudats` only (very low AP).

### D) Stronger adaptation attempt (ablation from improvement list)
- Config: `configs/experiments/noobb_iter2_adapt.yaml`
- Ran and interrupted at epoch 3 due runtime constraints; checkpoint saved under `runs/retfound_adapt/noobb_iter2_fgadr_uwf/`.
- Epoch losses observed before stop: 1.6634 -> 1.0514 -> 0.8734.

## Visual Outputs
- Prediction-only previews:
  - `eval/noobb_fgadr_base_uwf_preds/previews/`
  - `eval/noobb_fgadr_adapt_uwf_preds/previews/`
  - `eval/noobb_fgadr_adapt_uwf_tta_preds/previews/`
- GT+prediction overlays:
  - `eval/noobb_fgadr_base_uwf_preds/gt_overlays/`
  - `eval/noobb_fgadr_adapt_uwf_preds/gt_overlays/`
  - `eval/noobb_fgadr_adapt_uwf_tta_preds/gt_overlays/`
  - Shared-class-focused overlays: `eval/noobb_fgadr_adapt_uwf_tta_preds/gt_overlays_shared4_focus/`

## Reference best pipeline (with UWF supervision)
- `eval/main9_ensemble_best_preds/metrics_ap50_v2.json` -> `map50=0.2739`, `macro_f1=0.0548`.

## Conclusion from this ablation
- Removing UWF OBB supervision from train/val causes near-complete transfer collapse on UWF lesion detection.
- Representation alignment + TTA alone is not enough to recover useful localization/classification performance.
- This strongly supports keeping UWF-specific supervision (while improving its quality/noise handling), rather than dropping it entirely.
