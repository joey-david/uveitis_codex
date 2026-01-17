# Uveitis Codex

A machine learning project for localization of uveitis symptoms on ultra-wide-field fundus images.

This repo now implements the MVCAViT multi-view framework (CNN + ViT + cross-attention + PSO) for
classification + lesion localization, aligned to the Scientific Reports paper referenced in
`../nature_vit_cnn_dr_localization.pdf`.

## Datasets

Downloaders and organizers live in `datasets/` and are unchanged. See `datasets/datasets.md` for dataset notes.

Typical image roots:
- UWF-700: `datasets/uwf-700/Images/`
- DeepDRiD UWF: `datasets/deepdrid/ultra-widefield_images/ultra-widefield-training/Images/` and
  `datasets/deepdrid/ultra-widefield_images/ultra-widefield-validation/Images/`

## Local setup (CPU only)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Quick check:
```bash
pytest -q
```

## MVCAViT workflow (modular steps)

### 1) Build a multi-view manifest

Pair macula/optic images by filename suffix:
```bash
python scripts/build_multiview_manifest.py pairs \
  --root datasets/drtid/images \
  --macula-suffix _M --optic-suffix _O \
  --output manifests/drtid_train.jsonl
```
Output: `manifests/drtid_train.jsonl` (JSONL with `macula_path`, `optic_path`, `label`).

Or build from a CSV with columns `macula_path`, `optic_path`, `label`, `boxes` (JSON list of `[x1,y1,x2,y2]`):
```bash
python scripts/build_multiview_manifest.py csv \
  --csv manifests/drtid_train.csv \
  --output manifests/drtid_train.jsonl
```

### 2) Validate the manifest
```bash
python scripts/validate_manifest.py --manifest manifests/drtid_train.jsonl
```
Output: JSON with record count and any missing files or invalid boxes.

### 3) Train MVCAViT
```bash
python scripts/train_mvcavit.py \
  --train-manifest manifests/drtid_train.jsonl \
  --val-manifest manifests/drtid_val.jsonl \
  --output-dir runs/mvcavit \
  --num-classes 5 --num-boxes 10 \
  --use-pso
```
Outputs:
- `runs/mvcavit/last.pt` (latest checkpoint)
- `runs/mvcavit/best.pt` (best validation loss)
- `runs/mvcavit/metrics.json` (per-epoch losses)

### 4) Evaluate
```bash
python scripts/eval_mvcavit.py \
  --manifest manifests/drtid_val.jsonl \
  --checkpoint runs/mvcavit/best.pt
```
Output: JSON with accuracy and mean IoU.

### Uveitis single-view note
If you only have one view per image, use the same image for both views:
- Build a manifest with only `macula_path` and pass `--mirror-view` to training/eval.
If your labels are oriented boxes, pass `--box-format obb` to convert them to axis-aligned boxes.

For transfer learning from DR:
- Train on DR data first.
- Re-run `scripts/train_mvcavit.py` with `--pretrained runs/mvcavit/best.pt` on your uveitis manifest.

## Docker (GPU training)

See `docker.md` for a detailed, step-by-step guide with expected outputs.
