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
    *Supported: UWF-700, IDRiD, EyePACS, APTOS, DeepDRiD, FGADR, RFMiD, ODIR-5K.*
