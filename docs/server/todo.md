# TODO (server runbook)

## 0) Enter repo
```bash
cd /path/to/uveitis_codex
```

## 1) Build image
```bash
docker build -t uveitis-codex:latest .
```

## 2) Enter container
```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" -w /workspace \
  uveitis-codex:latest bash
```

## 3) Verify environment
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pytest -q
```
Expected: CUDA available prints True; pytest passes.

## 4) Prepare manifests
Prompt: "Where are the paired DR images and uveitis images located on disk?"

If paired DR images are in `datasets/drtid/images` with _M/_O suffix:
```bash
python scripts/build_multiview_manifest.py pairs \
  --root datasets/drtid/images \
  --macula-suffix _M --optic-suffix _O \
  --output manifests/drtid_train.jsonl
```

If uveitis labels are in CSV:
```bash
python scripts/build_multiview_manifest.py csv \
  --csv manifests/uveitis_train.csv \
  --output manifests/uveitis_train.jsonl
```

Validate:
```bash
python scripts/validate_manifest.py --manifest manifests/drtid_train.jsonl
python scripts/validate_manifest.py --manifest manifests/uveitis_train.jsonl
```

## 5) Train DR model (optional but recommended)
```bash
python scripts/train_mvcavit.py \
  --train-manifest manifests/drtid_train.jsonl \
  --val-manifest manifests/drtid_val.jsonl \
  --output-dir runs/mvcavit_dr \
  --num-classes 5 --num-boxes 10 \
  --use-pso
```

## 6) Fine-tune on uveitis
```bash
python scripts/train_mvcavit.py \
  --train-manifest manifests/uveitis_train.jsonl \
  --val-manifest manifests/uveitis_val.jsonl \
  --output-dir runs/mvcavit_uveitis \
  --pretrained runs/mvcavit_dr/best.pt \
  --mirror-view \
  --box-format obb
```

## 7) Evaluate
```bash
python scripts/eval_mvcavit.py \
  --manifest manifests/uveitis_val.jsonl \
  --checkpoint runs/mvcavit_uveitis/best.pt \
  --mirror-view \
  --box-format obb
```

## 8) Report back
Prompt: "Send back metrics.json, eval JSON output, and any manifest validation errors."
