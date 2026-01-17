## Docker workflow (GPU training)

This project is designed for remote GPU training inside Docker (no direct root GPU access required).

### 1) Build the image
```bash
docker build -t uveitis-codex:latest .
```
Output: Docker build logs ending with a successfully built image tagged `uveitis-codex:latest`.

### 2) Run a GPU container
```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" -w /workspace \
  uveitis-codex:latest bash
```
Output: an interactive shell inside the container at `/workspace`.

Alternative (docker compose):
```bash
docker compose run --rm --gpus all train bash
```

### 3) Build manifests inside the container
```bash
python scripts/build_multiview_manifest.py pairs \
  --root datasets/drtid/images \
  --macula-suffix _M --optic-suffix _O \
  --output manifests/drtid_train.jsonl
```
Output: `manifests/drtid_train.jsonl`.

### 4) Validate manifests
```bash
python scripts/validate_manifest.py --manifest manifests/drtid_train.jsonl
```
Output: JSON summary with record counts and any errors.

### 5) Train MVCAViT
```bash
python scripts/train_mvcavit.py \
  --train-manifest manifests/drtid_train.jsonl \
  --val-manifest manifests/drtid_val.jsonl \
  --output-dir runs/mvcavit \
  --use-pso
```
Outputs:
- `runs/mvcavit/last.pt`
- `runs/mvcavit/best.pt`
- `runs/mvcavit/metrics.json`

### 6) Evaluate
```bash
python scripts/eval_mvcavit.py \
  --manifest manifests/drtid_val.jsonl \
  --checkpoint runs/mvcavit/best.pt
```
Output: JSON with accuracy and mean IoU.

### Optional: single-view uveitis data
Use `--mirror-view` during training/evaluation if your manifest only has one image per record.
