# Datasets used by this repo

Run commands from the project root (`.../uveitis_codex`).

## What training code expects
- `datasets/uwf-700/Images`
- `datasets/deepdrid/ultra-widefield_images/ultra-widefield-training/Images`
- `datasets/deepdrid/ultra-widefield_images/ultra-widefield-validation/Images`
- `datasets/fgadr/...` (optional lesion masks)
- `datasets/eyepacs/...` (optional DR pretraining)
- your internal uveitis dataset folder (private)

## Current ingest flow
Use the bundled archive extractor:
```bash
python datasets/individual_dataset_processers/extract_uveitis_bundle.py \
  --zip datasets/raw/uveitis_datasets.zip \
  --out datasets
```
This creates/updates the canonical dataset folders above.

## Notes
- `datasets/raw/` is for archives/staging.
- `datasets/individual_dataset_downloaders/` are legacy per-dataset download scripts.
