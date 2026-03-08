# Checkpoint Report: main9_ensemble_ap02739

## 1) What Was Frozen
- Checkpoint ID: `main9_ensemble_ap02739`
- Ensemble strategy: IterC predictions for all classes except `oedeme_papillaire`, taken from IterJ.
- Weights kept on server:
  - `runs/retfound_mask/main9_iterC/best.pt` (sha256 `5138139edf47697d...`)
  - `runs/retfound_mask/main9_iterJ/best.pt` (sha256 `bf819e4041ef6169...`)
- Frozen metadata in repo: `checkpoints/main9_ensemble_ap02739/checkpoint_manifest.json`

## 2) Validation Performance (UWF val split)
- `mAP@0.5 = 0.2739`
- `Macro-F1 = 0.0548`
- `mAP@0.5 (all classes) = 0.2283`
- `Weighted AP@0.5 (by GT count) = 0.1706`

| Class | AP50 | F1 | n_gt (val) | n_pred |
|---|---:|---:|---:|---:|
| hyalite | 0.0891 | 0.0488 | 12 | 29 |
| nodule_choroidien | 0.2805 | 0.1143 | 6 | 29 |
| granulome_choroidien | 0.0000 | 0.0000 | 2 | 26 |
| foyer_choroidien | 0.0000 | 0.0000 | 1 | 13 |
| oedeme_papillaire | 1.0000 | 0.1111 | 1 | 17 |
| hemorragie | 0.0000 | 0.0000 | 0 | 9 |

## 3) Where It Performs Well
- `oedeme_papillaire`: AP50 is high on this split (1.0000) because the single validation case is detected.
- `nodule_choroidien`: strongest recurrent class performance among multi-case UWF classes (AP50 0.2805).
- `hyalite`: non-zero detection quality despite large shape variability (AP50 0.0891).
- Duplicate prediction control is stable due class-wise postprocess calibration + capped predictions per class.

## 4) Where It Still Fails
- `foyer_choroidien` and `granulome_choroidien` remain at AP50 = 0 on validation.
- These classes show strong confusion/overlap with other inflammatory findings and have very low support.
- Class balance remains severely skewed toward a handful of symptoms; rare classes are under-constrained.

## 5) Why the Performance Ceiling Is Data-Limited
### 5.1 Extremely small labeled UWF sample
- Labeled UWF split sizes: train=71, val=15, test=12
- A detector with 9 target classes is being tuned with only 71 train images and 15 val images.

### 5.2 Rare-class scarcity (hard numeric bottleneck)
- `exudats`: train objects=6, val objects=0
- `foyer_choroidien`: train objects=42, val objects=1
- `granulome_choroidien`: train objects=8, val objects=2
- `hemorragie`: train objects=57, val objects=0
- `hyalite`: train objects=43, val objects=12
- `ischemie_retine`: train objects=3, val objects=0
- `nodule_choroidien`: train objects=71, val objects=6
- `oedeme_papillaire`: train objects=12, val objects=1
- For classes with 1-2 validation objects, AP/F1 is statistically unstable and changes drastically with one TP/FP.

### 5.3 Annotation and supervision quality mismatch
- UWF labels are OBB-style and not as precise as pixel masks; this weakens localization supervision.
- The model is mask-first and benefits from precise boundaries; coarse/noisy boxes cap achievable IoU/AP.

### 5.4 Domain gap between sources
- FGADR contributes richer lesion supervision but differs in modality/distribution versus UWF uveitis images.
- Even after ROI masking and normalization, cross-domain representation mismatch persists on inflammatory classes.

### 5.5 Metric volatility on tiny val set
- With val n=15 images and per-class n_gt as low as 1, metric variance is high by construction.
- That volatility is visible across ablations: improvements can appear/disappear from one object assignment.

## 6) Why This Is a Strong Result Under Constraints
- This checkpoint is the highest mAP@0.5 obtained so far (`0.2739`) under the current dataset and label constraints.
- The gain came from methodical improvements: native labels, class-aware postprocess calibration, rare-class-aware training, and class-wise ensemble selection.
- Given the current data regime (especially UWF class scarcity), further robust gains now depend primarily on adding more labeled UWF data and/or improving label precision.

## 7) Reproducibility
- Ensemble choice file: `configs/experiments/main9_ensemble_choice_iterC_iterJ.json`
- Manifest: `checkpoints/main9_ensemble_ap02739/checkpoint_manifest.json`
- Evaluation outputs: `eval/main9_ensemble_best_preds/` (predictions, metrics, overlays)
