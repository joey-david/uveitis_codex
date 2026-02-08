# `qa_preproc_norm_to_regular.py`

## Purpose
Generate an on-disk QA bundle for the preprocessing pipeline on a small set of UWF-700 images:
- SAM2 ROI mask
- raw masked fundus-only image (outside fundus is black)
- crop
- normalization (including `reinhard_lab_ref` if configured)
- global resize + tiling

Outputs are saved in a single folder for quick inspection.

## CLI
```bash
python scripts/qa_preproc_norm_to_regular.py \
  --config configs/stage0_preprocess.yaml \
  --n 20 \
  --out eval/preproc_norm_qa_uwf20 \
  --overwrite
```

## Writes
- `eval/preproc_norm_qa_uwf20/overlays/*.png` raw + ROI boundary
- `eval/preproc_norm_qa_uwf20/masks/*.png` binary mask
- `eval/preproc_norm_qa_uwf20/masked_raw/*.png` raw masked fundus-only
- `eval/preproc_norm_qa_uwf20/norm/*.png` normalized fundus-only
- `eval/preproc_norm_qa_uwf20/global_1024/*.png` global resized
- `eval/preproc_norm_qa_uwf20/tiles/<image_key>/*.png` tiles
- `eval/preproc_norm_qa_uwf20/triptychs/*.png` side-by-side (overlay | masked raw | normalized)
- `eval/preproc_norm_qa_uwf20/selected_images.json` selection + per-image metadata

## Notes
- Picks a diverse sample across `datasets/uwf-700/Images/*` subfolders by default.
- Intended for "does this look right?" validation before running full Stage 0 preprocessing.

