# Improvement Loop (Noise-Aware + Ablations)

## Goal
Push performance back above the previous peak (`map50=0.2739`) and test at least 4 additional paths after noise-aware training.

## Runs and Results (UWF val, AP50)

| Variant | map50 | macro_f1 | Notes |
|---|---:|---:|---|
| `main9_ensemble_best_preds` (previous peak) | 0.2739 | 0.0548 | class-wise ensemble (iterC + iterJ) |
| `main9_iterK_noise_preds` | 0.1337 | 0.0458 | first noise-aware run |
| `main9_iterL_noise_sampler_preds` | 0.2359 | 0.0518 | noise-aware + sampler/class-weights/focal |
| `main9_iterL_noise_sampler_tta_preds` | 0.0500 | 0.0222 | multiscale+hflip TTA (degraded) |
| `main9_iterL_noise_sampler_union_preds` | 0.2337 | 0.0554 | union-component extraction |
| `main9_iterL_noise_sampler_sweep/nms0p1_k1` | 0.2366 | 0.0693 | postprocess sweep best (`max_preds_per_class=1`) |
| `main9_iterM_noise_1024_preds` | 0.1298 | 0.0571 | 1024 training; recovered granulome AP |
| `main9_iterN_noise_rarefocus_preds` | 0.2337 | 0.0544 | rare-focused reweighting |
| `main9_ensemble_plus_iterM_preds` | **0.3076** | **0.0673** | **new best**: previous peak ensemble + `iterM` for `granulome_choroidien` |

## What improved
- The new best (`eval/main9_ensemble_plus_iterM_preds/metrics_ap50_v2.json`) increases AP50 from **0.2739 -> 0.3076**.
- Gain source is class transfer for `granulome_choroidien`:
  - old AP50: `0.0000`
  - new AP50: `0.1683`
- Other strong classes remained unchanged (`hyalite`, `nodule_choroidien`, `oedeme_papillaire`).

## Key Artifacts
- New best predictions: `eval/main9_ensemble_plus_iterM_preds/predictions.jsonl`
- New best metrics: `eval/main9_ensemble_plus_iterM_preds/metrics_ap50_v2.json`
- Ensemble mapping used: `configs/experiments/main9_ensemble_plus_iterM_choice.json`
- Visual overlays: `eval/main9_ensemble_plus_iterM_preds/overlays`

## Repro Commands
```bash
# Build new best class-wise ensemble
uv run python scripts/ensemble_native_predictions.py \
  --pred-a eval/main9_ensemble_best_preds/predictions.jsonl \
  --pred-b eval/main9_iterM_noise_1024_preds/predictions.jsonl \
  --class-source-json configs/experiments/main9_ensemble_plus_iterM_choice.json \
  --out-jsonl eval/main9_ensemble_plus_iterM_preds/predictions.jsonl \
  --out-meta eval/main9_ensemble_plus_iterM_preds/meta.json

# Evaluate
uv run python scripts/eval_native_detection.py \
  --gt-index labels_native_main9_fastiter/uwf700_val_global.jsonl \
  --pred-jsonl eval/main9_ensemble_plus_iterM_preds/predictions.jsonl \
  --out eval/main9_ensemble_plus_iterM_preds/metrics_ap50_v2.json \
  --iou 0.5 --score 0.0
```
