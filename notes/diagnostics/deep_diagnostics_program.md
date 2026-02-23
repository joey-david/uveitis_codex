# Deep Diagnostics Program: UWF Detection Pipeline

## Objective
Raise end-to-end UWF performance by identifying dominant failure points and fixing them with measured gains.

Primary metric:
- `UWF val mAP50-95` on staged system (main9 OBB + vascularite module)

Secondary metrics:
- `mAP50`
- Per-class AP/recall (focus on weakest classes)
- Duplicate OBB rate (same class, high overlap, same lesion)
- Calibration quality (precision/recall vs confidence)

## Baseline to Beat
- Main model: `out/weights/yolo_obb_main9_mmproxy_contrastive_best.pt`
- Vascularite model: `out/weights/yolo_obb_vascularite_best.pt`
- Data:
  - Main: `out/yolo_obb/uwf700_tiles_main9/data.yaml`
  - Vascularite: `out/yolo_obb/uwf700_tiles_vascularite_only/data.yaml`

## Subagent Workstreams
We run four concurrent workstreams ("subagents"), each writing to `eval/diagnostics/` and `notes/diagnostics/`.

### Subagent A: Data + Label Integrity
- Validate OBB geometry validity and clipping.
- Audit class frequencies and train/val mismatch.
- Detect leakage/near-duplicates across splits.
- Produce stratified GT overlays for manual checks.

### Subagent B: Preprocessing Diagnostics
- Verify SAM mask quality and area statistics.
- Validate that normalization excludes non-fundus black zones.
- Run preprocess ablations: mask-only vs mask+norm variants.
- Measure class-wise impact from preprocess variants.

### Subagent C: Main Detector Hypothesis Sweeps
- Run high-signal ablations (single-variable changes):
  - resolution
  - augmentation intensity
  - freeze depth
  - loss/optimizer schedule tweaks
  - class-balanced sampling variants
- Re-run top candidates with fixed seeds for stability.

### Subagent D: Postprocess + Vascularite Integration
- Diagnose duplicate OBB clusters.
- Compare global polygon NMS / soft-NMS-like suppression settings.
- Evaluate vascularite module alone and integrated with main.
- Quantify integration tradeoff on overall mAP and per-class recall.

## Hypothesis Matrix
H1. Main misses are dominated by small/elongated lesions under current scale and augmentation.

H2. Duplicate same-class OBBs reduce precision and AP; tuned global suppression can recover AP.

H3. Some normalization settings suppress weak lesion contrast; post-mask photometric settings need retuning.

H4. Long-tail class sparsity still harms recall; balanced mixtures/weighting need tighter calibration.

H5. Vascularite specialization improves full-system performance when isolated from the main detector.

## Diagnostics Deliverables
- `eval/diagnostics/baseline_repro/*.json`
- `eval/diagnostics/data_qc/*.json`
- `eval/diagnostics/preproc_ablation/*.json`
- `eval/diagnostics/hypothesis_sweeps/*.json`
- `eval/diagnostics/dedup/*.json`
- `eval/diagnostics/combined_previews/*.png`
- `notes/diagnostics/failure_points.md`
- `notes/diagnostics/diagnostic_runbook.md`

## Acceptance Criteria
- Improve staged-system `mAP50-95` by at least `+0.04` vs baseline.
- Improve recall on worst two symptom classes by at least `+0.10` absolute.
- Reduce duplicate OBB index by at least `40%` with no significant recall collapse.

## Execution Rules
- Every run is config-backed and reproducible.
- Keep artifacts compact; prune failed heavy checkpoints/logs.
- Never change label policy:
  - main detector = FGADR classes + top-6 UWF classes (excluding vascularite)
  - vascularite handled by separate module.
