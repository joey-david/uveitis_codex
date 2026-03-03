# UWF Overfit Push (Performance Ceiling)

## Objective
Deliberately overfit on the UWF set to maximize localization/classification on the current UWF validation split.

## Important Caveat
The specialist run `main9_iterO_uwf_overfit` was trained on:
- `labels_native_main9_fastiter/uwf700_all_global.jsonl`

This includes train+val+test images. The resulting metrics are therefore a **ceiling-style overfit estimate**, not an unbiased generalization score.

## What Was Trained
- Train config: `configs/experiments/main9_iterO_uwf_overfit_mask.yaml`
- Checkpoint: `runs/retfound_mask/main9_iterO_uwf_overfit/best.pt`
- Key settings: full unfreeze (`freeze_blocks: 0`), heavy class balancing, focal loss, UWF noise-aware weighting.

## Base Overfit Result
- Predictions: `eval/main9_iterO_uwf_overfit_preds/predictions.jsonl`
- Metrics: `eval/main9_iterO_uwf_overfit_preds/metrics_ap50_v2.json`
- AP50: `0.3347`

## Key Discovery
For `foyer_choroidien`, the model had strong local activation but default low thresholds produced giant components and bad boxes.

Fix used:
- class-specific postprocess (`foyer_choroidien`):
  - `threshold: 0.5`
  - `min_area_px: 500`
  - `close_kernel: 0`
- For `granulome_choroidien`: `threshold: 0.5`
- inference cap: `max_preds_per_class: 2`

Configs:
- `configs/experiments/main9_iterO_overfit_focusfg_postprocess.json`
- `configs/experiments/main9_iterO_overfit_focusfg_infer.yaml`

## Best Combined Result
Class-wise ensemble:
- A: `eval/main9_ensemble_plus_iterO_preds/predictions.jsonl`
- B: `eval/main9_iterO_overfit_focusfg_preds/predictions.jsonl`
- class source: `configs/experiments/main9_ensemble_plus_iterO_focusfg_choice.json`
  - `foyer_choroidien` -> B
  - `granulome_choroidien` -> B

Output:
- `eval/main9_best_overfit_preds/predictions.jsonl`
- `eval/main9_best_overfit_preds/metrics_ap50_v2.json`

Best AP50:
- **`0.6739`**

Per-class AP50 (val):
- `foyer_choroidien`: `1.0000`
- `granulome_choroidien`: `1.0000`
- `hyalite`: `0.0891`
- `nodule_choroidien`: `0.2805`
- `oedeme_papillaire`: `1.0000`

## Visual QA
- Overlays: `eval/main9_best_overfit_preds/overlays`
- Montage: `eval/main9_best_overfit_preds/overlays/montage_5x3.png`

## Frozen Artifact
- `checkpoints/main9_overfit_ap06739`

Contains metrics, configs, predictions, ensemble mapping, and manifest.
