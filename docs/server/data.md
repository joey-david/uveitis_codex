# Data + Manifests

## Folder expectations
- datasets/uwf-700/Images
- datasets/deepdrid/ultra-widefield_images/ultra-widefield-training/Images
- datasets/deepdrid/ultra-widefield_images/ultra-widefield-validation/Images
- datasets/uvelabels/... (your uveitis images and boxes)

## Build multi-view manifests
### Paired views by suffix
```bash
python scripts/build_multiview_manifest.py pairs \
  --root datasets/drtid/images \
  --macula-suffix _M --optic-suffix _O \
  --output manifests/drtid_train.jsonl
```

### From CSV (recommended for boxes)
CSV columns: macula_path, optic_path, label, boxes
- boxes is JSON list of [x1,y1,x2,y2]

```bash
python scripts/build_multiview_manifest.py csv \
  --csv manifests/drtid_train.csv \
  --output manifests/drtid_train.jsonl
```

### Single-view (uveitis)
- Build manifest with only macula_path.
- Use --mirror-view in training/eval.

## Validate manifest
```bash
python scripts/validate_manifest.py --manifest manifests/drtid_train.jsonl
```
Expected: JSON with record count and zero errors.
