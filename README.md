# Uveitis Codex

A machine learning project for localization of uveitis symptoms on ultra-wide-field fundus images.

## Environment Setup

This project uses Docker for reproducibility.

```bash
docker-compose build
docker-compose up -d
docker-compose exec dataset-prep bash
```

## Datasets

We use a collection of public retinal datasets for pretraining and domain adaptation.

### Structure
- **Raw Data**: `datasets/raw/` (Download destination)
- **Processed**: `datasets/processed/` (Standardized format: `images/train/*.jpg`, `labels/...`)

### Workflow

1.  **Download**: Use the download script to fetch data.
    ```bash
    # Check status and instructions (Dry Run)
    python datasets/download_datasets.py --dry-run
    
    # Download everything (Direct + Kaggle)
    python datasets/download_datasets.py
    ```
    *Note: Kaggle datasets require `kaggle.json` in root/`~/.kaggle`. Some datasets (FGADR) require manual download to `datasets/raw`.*

2.  **Standardize**: Convert all raw data into the unified project structure.
    ```bash
    python datasets/standardize_datasets.py
    ```
    *Supported: UWF-700, DeepDRiD, FGADR, EyePACS, Uveitis-DISP (manual).*

## Stage 1: RETFound MAE Adaptation (UWF)

Prereqs:
- UWF-700 images: `datasets/uwf-700/Images/`
- DeepDRiD UWF images: `datasets/deepdrid/ultra-widefield_images/ultra-widefield-training/Images/` and `datasets/deepdrid/ultra-widefield_images/ultra-widefield-validation/Images/`

Run these 3 commands in order:

1) Login for gated RETFound weights (checkpoint will download into `~/.cache/huggingface/` on first use).
```bash
huggingface-cli login
```

2) Build the unlabeled manifest and run Stage-1 MAE adaptation (outputs `manifests/stage1_unlabeled.jsonl` and `runs/stage1/mae_adapt_last.pth`, `runs/stage1/encoder_adapted.pth`).
```bash
python scripts/build_manifest.py unlabeled \
  --roots datasets/uwf-700/Images \
          datasets/deepdrid/ultra-widefield_images/ultra-widefield-training/Images \
          datasets/deepdrid/ultra-widefield_images/ultra-widefield-validation/Images \
  --output manifests/stage1_unlabeled.jsonl \
  && python scripts/stage1_adapt_mae.py \
    --manifest manifests/stage1_unlabeled.jsonl \
    --output-dir runs/stage1
```

DDP example (optional, multi-GPU):  
`torchrun --nproc_per_node=4 scripts/stage1_adapt_mae.py --manifest manifests/stage1_unlabeled.jsonl --output-dir runs/stage1`

3) Build UWF-700 splits and run the before/after linear probe (outputs `manifests/uwf700/*.jsonl`, `manifests/uwf700/labels.json`, `runs/linear_probe/results.json`).
```bash
python scripts/build_manifest.py uwf700 --images-dir datasets/uwf-700/Images --out-dir manifests/uwf700 \
  && python scripts/eval_linear_probe.py \
    --train manifests/uwf700/train.jsonl \
    --val manifests/uwf700/val.jsonl \
    --test manifests/uwf700/test.jsonl \
    --adapted runs/stage1/encoder_adapted.pth \
    --output runs/linear_probe/results.json
```

## Docker (GPU training)

Build the image (once):
```bash
docker build -t uveitis-codex:latest .
```

Run an interactive container with GPU access (outputs and checkpoints stay in your working tree):
```bash
docker run --rm -it --gpus all -v "$PWD:/workspace" -w /workspace uveitis-codex:latest bash
```

Inside the container, run the same 3 commands above. HuggingFace caches to `/root/.cache/huggingface/`, manifests go to `manifests/`, and checkpoints/results to `runs/`.
