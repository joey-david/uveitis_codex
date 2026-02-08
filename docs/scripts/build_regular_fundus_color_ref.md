# `build_regular_fundus_color_ref.py`

## Purpose
Compute a *regular fundus* color reference (LAB mean/std and RGB mean/std) using ROI-masked, non-black pixels only.

This is used by `normalize.method: reinhard_lab_ref` to normalize UWF fundus colors toward regular fundus statistics.

Datasets treated as "regular fundus" by default:
- `eyepacs`
- `fgadr`
- `deepdrid_regular`

## CLI
```bash
python scripts/build_regular_fundus_color_ref.py \
  --config configs/stage0_preprocess.yaml \
  --per-dataset 50 \
  --out preproc/ref/regular_fundus_color_stats.json
```

## Reads
- Manifests listed in `configs/stage0_preprocess.yaml` (must exist).
- Regular fundus images from the datasets in those manifests.
- ROI config (`roi.*`) to ensure stats are computed on the same ROI definition used by preprocessing.

## Writes
- `preproc/ref/regular_fundus_color_stats.json`

Contains:
- `lab_mean`, `lab_std`
- `rgb_mean`, `rgb_std`
- `n_images_used`, `n_pixels`

## Notes
- Stats are computed on ROI pixels only (and eroded by `normalize.stats_erode_px`).
- Pure black pixels are excluded (fundus borders and already-masked regions).
- Sampling is *per dataset* to avoid Eyepacs dominating the reference.

