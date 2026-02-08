# `uveitis_pipeline.reports`

Reporting utilities for dataset, preprocessing, training, and ablation.

## Functions

| Function | Description |
|---|---|
| `report_dataset(manifest_paths, coco_paths, out_json)` | Aggregates dataset/split/label-format counts and COCO annotation stats. |
| `report_preproc(manifest_path, preproc_root, out_dir, sample_n=24)` | Creates raw-vs-norm grids, channel histograms, ROI area summary. |
| `report_training(run_dir, out_json, out_png)` | Summarizes training metrics and renders loss/mAP proxy curves. |
| `ablate_preproc(pred_dir_a, pred_dir_b, out_json)` | Compares prediction counts across two prediction directories. |
