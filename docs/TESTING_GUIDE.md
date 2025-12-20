# Testing Specific Parts

## Test MTSN training quickly
- Reduce epochs/batch size in `config.yaml` under `mtsn.training`.
- Run: `python -m src.mtsn.train_mtsn`
- Inspect `metrics/mtsn_train.csv` for AUC/PR and timing.

## Test OVV on a few images
- Ensure `cfg.mtsn.paths.model_save_path` points to a trained MTSN.
- Point `ovv.paths.target_img_dir` to a small folder.
- Run: `python -m src.ovv.runner`
- Inspect `metrics/ovv_labeling.csv` and `metrics/ovv_summary.json`.

## Test GACNN with a tiny graph set
- Build graphs for a subset (copy a handful of images to a temp folder and set `gacnn.paths.*` to those).
- Run: `python -m src.gacnn.make_graphs`, then `python -m src.gacnn.training`.
- Inspect `metrics/gacnn_train.csv` and `metrics/gacnn_obb_summary_epoch*.json`.
