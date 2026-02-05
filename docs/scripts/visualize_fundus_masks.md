# `datasets/uwf-700/visualize_fundus_masks.py`

## Purpose
Interactive QA viewer for UWF fundus masks (prompted SAM for `uwf700`, threshold fallback if unavailable).

## CLI
```bash
python datasets/uwf-700/visualize_fundus_masks.py \
  --images datasets/uwf-700/Images/Uveitis \
  --config configs/stage0_preprocess.yaml \
  --max-images 100
```

## Views
- Raw image with mask boundary overlay
- Masked fundus-only image
- Binary mask

## Controls
- Keyboard: `left/a` previous, `right/d` next
- Buttons: `< Prev`, `Next >`

## Reads
- UWF image directory
- `roi` + `roi.sam` config in `configs/stage0_preprocess.yaml`

## Notes
- Uses the same `compute_roi_mask(...)` path as stage preprocessing.
- Prints fallback notice if SAM checkpoint/import is unavailable.
