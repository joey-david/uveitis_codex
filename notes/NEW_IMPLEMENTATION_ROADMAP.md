# New Implementation Roadmap

> Implementation rule: at each step of each stage, keep running notes on what was done, obstacles encountered, solutions applied, and current status/progress (especially after context compactions).

## Context
Goal: maximize lesion localization + classification performance on UWF images (focused label space), while keeping the pipeline modular, debuggable, and reproducible.

Core tools in this implementation:
- SAM2 for fundus ROI masking (remove machinery/eyelids/background first).
- ROI-aware normalization (stats computed only on non-black, in-mask pixels).
- RETFound as the main representation backbone, then mask-first localization heads.
- Mask-to-OBB/box conversion + class refinement for final outputs.

Known issues from previous iterations to actively avoid:
- Domain gap between FGADR-style regular fundus and UWF appearance hurt transfer.
- Noisy/coarse UWF OBB supervision reduced detector quality.
- Duplicate overlapping predictions for one lesion (postprocess/NMS issue).
- Background-heavy tiles and black-border pixels biasing training/normalization.

1. Freeze target label space to FGADR-overlap classes + top-6 UWF classes; split vascularite into its own branch.
2. Keep SAM2 as baseline fundus ROI extraction; preprocess only inside ROI (color/contrast stats ignore black background).
3. Build high-res (1024+ tile) training views with overlap and strict GT alignment QA.
4. Adapt RETFound on masked ROI crops (contrastive/domain adaptation to FGADR style).
5. Train mask-first lesion heads (precise FGADR masks + weak UWF supervision), then derive boxes/OBB from masks at inference.
6. Add class refinement on ROI proposals, pseudo-label loop on UWF, and vascularite-specific fusion branch.
7. Calibrate thresholds/NMS per class and validate with visual overlays + per-class error analysis.

## Implementation Log
- 2026-02-25 (Stage 1 pivot):
  - Obstacle: COCO export was too coarse for our objective and lost shape precision.
  - Decision: replaced COCO as primary label format with native per-image/per-tile polygon records (`labels_native`).
  - Action: implemented `src/uveitis_pipeline/labels_native.py` and rewired `scripts/stage0_build_labels.py` to build native labels directly from FGADR masks + UWF OBBs.
  - Status: native labeling stage is now the default path; class-space filtering uses `configs/active_label_space.yaml`.
- 2026-02-25 (Stage 2 prep):
  - Obstacle: repo cleanup removed reporting/training plumbing.
  - Action: added new minimal modules for reports (`src/uveitis_pipeline/reports.py`), native datasets (`src/uveitis_pipeline/native_dataset.py`), and RETFound mask-first modeling (`src/uveitis_pipeline/retfound_mask.py`).
  - Status: stage scripts for adaptation/training/inference/calibration were restored in native format (`scripts/stage4_*` to `scripts/stage7_*`).
- 2026-02-25 (Stage validation smoke checks):
  - Commands run successfully in `uveitis-codex:latest` container:
    - `scripts/stage0_build_manifest.py` on `configs/stage0_manifest_smoke.yaml`.
    - `scripts/stage0_preprocess.py` on tiny configs (`preproc_smoke`, then labeled subset).
    - `scripts/stage0_build_labels.py` in native mode on labeled subset (`labels_native_labeled_smoke`).
    - `scripts/qa_native_labels.py` for visual overlay sanity checks.
    - `scripts/stage4_adapt_retfound.py` (1-epoch smoke).
    - `scripts/stage5_train_mask_head.py` (1-epoch smoke, after fixing pos-embed mismatch loading).
    - `scripts/stage6_infer_mask_to_obb.py` and `scripts/stage7_calibrate_thresholds.py` smoke runs.
  - Obstacle: adapted checkpoint pos-embed shape mismatch when moving from 224 adaptation to larger train size.
  - Fix: added interpolating loader `load_encoder_state(...)` in `src/uveitis_pipeline/retfound_mask.py`.
- 2026-02-25 (Current loop: train/eval/adapt iterations):
  - Prepared GPU+docker execution path and found critical runtime detail: CUDA works only with `--runtime=nvidia --gpus all` (plain `--gpus all` gave `torch.cuda.is_available=False`).
  - Added local SAM2 config (`configs/sam2.1/sam2.1_hiera_b+.yaml`) and downloaded SAM2.1 base+ checkpoint (`models/sam2/sam2.1_hiera_base_plus.pt`).
  - Built full fast manifests and generated `manifests/uwf700_labeled.jsonl` (98 labeled UWF images).
  - Computed regular fundus color reference on 150 sampled images (50 per regular dataset family): `preproc/ref/regular_fundus_color_stats_50.json`.
  - Started iterative training data preprocessing (`preproc_main9_fastiter`) on FGADR-train700 + UWF-labeled with SAM2-first ROI + ROI-only Reinhard normalization.
- 2026-02-25 (Iterative training diagnostics + adaptation loops):
  - Implemented native detection evaluator: `scripts/eval_native_detection.py` (AP/F1 on native GT/preds).
  - Iteration B/C/D/E cycles executed with calibration + inference sweeps.
  - Main improvement came from:
    - sparse-positive-aware mask loss (`src/uveitis_pipeline/retfound_mask.py`),
    - global-image training mix (IterC),
    - postprocess cap per class (`max_preds_per_class=2`) in `scripts/stage6_infer_mask_to_obb.py`.
  - Best current checkpoint: `runs/retfound_mask/main9_iterC/best.pt`.
  - Best current metrics (UWF val, native eval @IoU 0.5):
    - `map50=0.1073` (classes with GT in split),
    - `macro_f1=0.0559`.
  - Artifacts saved under `eval/main9_best_preds_v2/` and report in `notes/ITERATION_REPORT_2026-02-25.md`.
- 2026-02-25 (Inference/postprocess ablation pass):
  - Added detection-level per-class postprocess calibration (`scripts/stage7_calibrate_detection_postprocess.py`) and class-aware postprocess override loading in stage6 (`class_postprocess_json`).
  - Result: iterC + detection-level calibration improved unweighted val `map50` from `0.1073` to `0.1239` (`eval/main9_iterC_detcalib_preds/metrics_ap50_v2.json`).
  - Added optional inference TTA (scale/flip) and optional union-component mode in `stage6`; TTA/union did not beat per-class mode on map.
- 2026-02-25 (Training ablation pass: iterG/iterJ):
  - Added sampler controls (`sampler_mode`, `sampler_power`, `empty_sample_weight`) and optional auto class-weighting in stage5 + weighted loss path in `masked_bce_dice_loss`.
  - IterG (1024 + global+UWF tiles) reduced weighted quality (`map50=0.0746`), rejected.
  - IterJ (short fine-tune from iterC with class-weighted loss) gave high unweighted `map50=0.2337` but collapsed hyalite (`AP=0`), so not selected as standalone best.
- 2026-02-25 (Model ensembling):
  - Brute-forced class-wise merge of iterC-detcalib and iterJ predictions on val.
  - Best merge keeps iterC for all classes except `oedeme_papillaire` from iterJ.
  - Ensemble metrics: `map50=0.2739`, `weighted_ap50=0.1706`, `macro_f1=0.0548`.
  - Artifacts: `eval/main9_ensemble_best_preds/predictions.jsonl`, `eval/main9_ensemble_best_preds/metrics_ap50_v2.json`, previews under `eval/main9_ensemble_best_preds/previews`.
