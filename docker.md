## Docker Runbook (Remote GPU)

This runbook is the command-by-command reference for running the full pipeline inside Docker on a remote GPU host.

## 1) General Docker Environment Setup

### 1.0 GPU smoke test (host shell, before heavy builds)

This host setup requires `--runtime=nvidia` (using `--gpus all` alone can yield `Failed to initialize NVML: Unknown Error`).

Minimal NVML check:

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi -L
```

PyTorch CUDA check (matches our base image CUDA stack):

```bash
docker run --rm --runtime=nvidia --gpus all pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime \
  python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

### 1.1 Build image (host shell)

```bash
docker build -t uveitis-codex:latest .
```

What it takes in:
- `Dockerfile`
- `requirements.txt`
- repository source code

What it produces:
- local Docker image `uveitis-codex:latest`

What to expect:
- standard Docker build logs
- final success line like `Successfully tagged uveitis-codex:latest`

### 1.2 Launch interactive GPU container (host shell)

```bash
docker run --rm -it --runtime=nvidia --gpus all \
  --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD:/workspace" \
  -w /workspace \
  uveitis-codex:latest bash
```

Alternative:

```bash
docker compose run --rm train bash
```

If you want artifacts written to the mounted repo to be owned by your host user (recommended):

```bash
docker compose run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp train bash
```

What it takes in:
- built image
- mounted repo (`$PWD -> /workspace`)
- GPU runtime

What it produces:
- interactive shell in container at `/workspace`

What to expect:
- bash prompt inside container
- from this point onward, run normal `python ...` commands directly

### 1.3 Verify environment once inside container

```bash
python - <<'PY'
import torch, sys
print({'python': sys.version.split()[0], 'cuda_available': torch.cuda.is_available(), 'gpu_count': torch.cuda.device_count()})
PY
```

What it takes in:
- container Python + torch install

What it produces:
- environment check dict

What to expect:
- `cuda_available: True` and `gpu_count >= 1` on GPU host

### 1.4 Download SAM2 checkpoint once (inside container)

```bash
mkdir -p models/sam2
curl -L --fail -o models/sam2/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

What it takes in:
- internet access

What it produces:
- `models/sam2/sam2.1_hiera_base_plus.pt`

What to expect:
- download progress and saved checkpoint file

Optional quick QA of fundus masks:

```bash
python datasets/uwf-700/visualize_fundus_masks.py \
  --images datasets/uwf-700/Images/Uveitis \
  --config configs/stage0_preprocess.yaml \
  --max-images 30
```

What to expect:
- interactive viewer with three panes: boundary overlay, masked fundus-only image, binary mask

Non-interactive QA on 50 images (writes overlays/masks to disk):

```bash
python scripts/qa_fundus_mask_sam2.py --images datasets/uwf-700/Images --n 50 --out-dir eval/sam2_fundus_qa
```

Outputs:
- `eval/sam2_fundus_qa/overlays/*.png` (raw + mask boundary)
- `eval/sam2_fundus_qa/masked/*.png` (fundus-only RGB)
- `eval/sam2_fundus_qa/masks/*.png` (binary mask)

## 2) Pipeline Steps and Substages (Run Inside Docker Env)

### Stage 0.0 — Dataset ingestion + manifest

```bash
python scripts/stage0_build_manifest.py --config configs/stage0_manifest.yaml
```

Takes in:
- dataset roots configured in `configs/stage0_manifest.yaml`

Produces:
- `manifests/{dataset}.jsonl`
- `manifests/{dataset}.csv`
- `splits/stage0_0.json`

Expect:
- `Wrote manifests to manifests`
- split + label-format + class-count summaries per dataset

### Stage 0.1 — Retina ROI mask + crop

```bash
python scripts/stage0_preprocess.py --config configs/stage0_preprocess.yaml
```

Takes in:
- manifests listed in `configs/stage0_preprocess.yaml`
- UWF SAM2 settings from `roi.sam2` (fallback to threshold if unavailable)

Produces:
- `preproc/roi_masks/{image_id}.png`
- `preproc/crops/{image_id}.png`
- `preproc/crop_meta/{image_id}.json`

Expect:
- `Preprocessing complete`
- metrics dict including ROI quality rates

### Stage 0.2 — Retina-only photometric normalization

Command:
- same `stage0_preprocess.py` command as Stage 0.1

Takes in:
- cropped image + ROI/stat masks
- optional regular-fundus reference stats if using `normalize.method: reinhard_lab_ref`:
  - build with `python scripts/build_regular_fundus_color_ref.py --per-dataset 50`
  - writes `preproc/ref/regular_fundus_color_stats.json` (path set by `normalize.ref.stats_path`)

Produces:
- `preproc/norm/{image_id}.png`
- `preproc/norm_meta/{image_id}.json`

Expect:
- outside-fundus pixels masked out in normalized image
- normalization metadata with `method`, channel `mean/std`

### Stage 0.3 — Canonical resizing + tiling

Command:
- same `stage0_preprocess.py` command as Stage 0.1

Takes in:
- normalized crop outputs

Produces:
- `preproc/global_1024/{image_id}.png`
- `preproc/tiles/{image_id}/{tile_id}.png`
- `preproc/tiles_meta/{image_id}.json`
- `preproc/verify/preprocess_metrics.json`

Expect:
- tile count distribution in metrics
- reconstruction sanity error reported (`avg_reconstruction_abs_error`)

### Stage 0.4 — Label conversion + harmonization

```bash
python scripts/stage0_build_labels.py --config configs/stage0_labels.yaml
```

Takes in:
- manifests + split file
- `configs/class_map.yaml`
- preprocessing metadata from Stage 0.1-0.3

Produces:
- `labels_coco/{dataset}_{split}.json`
- `labels_coco/{dataset}_{split}_tiles.json`
- `labels_coco/summary.json`
- `labels_debug/{dataset}_{split}/*.png`

Expect:
- per-dataset/per-split summary logs (`num_images`, `num_annotations`, class counts)

### Stage 1.0 — Baseline detector training scaffold

```bash
python scripts/train_detector.py --config configs/train_overfit10.yaml
```

Takes in:
- COCO train/val labels from Stage 0.4
- model/training settings from config

Produces:
- `runs/overfit_10/config.yaml`
- `runs/overfit_10/checkpoints/{epoch_*.pth,best.pth,last.pth}`
- `runs/overfit_10/metrics.jsonl`
- `runs/overfit_10/val_report.json`

Expect:
- epoch checkpoints written as training progresses
- validation report emitted at end

### Stage 1.1 — Inference + tile-to-global merge

```bash
python scripts/infer_detector.py --config configs/infer_uveitis_ft.yaml
```

Takes in:
- checkpoint in config (`model.checkpoint`)
- preprocessed tiles + tile metadata

Produces:
- `preds/uveitis_ft/{image_id}.json`
- `preds_vis/uveitis_ft/{image_id}.png` (when detections exist)

Expect:
- merged per-image predictions after class-wise global NMS

### Stage 2.0 — FGADR box dataset build

```bash
python scripts/stage0_build_labels.py --config configs/stage0_labels.yaml
```

Takes in:
- FGADR masks + class map

Produces:
- `labels_coco/fgadr_{train,val,test}.json`
- `labels_coco/fgadr_{train,val,test}_tiles.json`

Expect:
- FGADR class counts visible in summary logs and `labels_coco/summary.json`

### Stage 2.1 — Train detector on FGADR

```bash
python scripts/train_detector.py --config configs/train_fgadr.yaml
```

Takes in:
- FGADR tile COCO labels

Produces:
- `runs/fgadr_pretrain/...`

Expect:
- checkpoints and metrics under `runs/fgadr_pretrain`

### Stage 3.0 — UWF uveitis dataset build (HBB)

```bash
python scripts/stage0_build_labels.py --config configs/stage0_labels.yaml
```

Takes in:
- UWF OBB labels + class map + preprocess metadata

Produces:
- `labels_coco/uwf700_{train,val,test}.json`
- `labels_coco/uwf700_{train,val,test}_tiles.json`

Expect:
- UWF class counts in summary logs and `labels_coco/summary.json`

### Stage 3.1 — Fine-tune on UWF uveitis

```bash
python scripts/train_detector.py --config configs/train_uveitis_ft.yaml
```

Takes in:
- UWF tile COCO labels
- optional FGADR init checkpoint (`training.init_checkpoint`)

Produces:
- `runs/uveitis_ft/...`
- `runs/uveitis_ft/val_report.json`

Expect:
- val report includes per-class sensitivity at configured FP/image targets

### Stage 4.A (optional) — Continue MAE pretraining

```bash
python scripts/stage4_continue_mae.py --retfound-dir ../RETFound --data-path preproc/global_1024 --run
```

Takes in:
- RETFound repo with `main_pretrain.py`
- preprocessed image directory

Produces:
- MAE outputs under configured `--output-dir`

Expect:
- if script missing, prints structured `skipped` status instead of failing pipeline

### Stage 4.B (optional) — Pseudo-label expansion

```bash
python scripts/stage4_pseudo_label_expand.py \
  --base-coco labels_coco/uwf700_train_tiles.json \
  --pred-dir preds/uveitis_ft \
  --out-coco pseudo_labels/uveitis_ft/train_plus_pseudo.json
```

Takes in:
- base COCO labels
- model predictions

Produces:
- expanded pseudo-label COCO file

Expect:
- printed summary with image/annotation/pseudo counts

### Stage 4.C (optional) — Rotated head

Status:
- not implemented in current codebase

## 3) Bottleneck/QA Hooks (Run Inside Docker Env)

### Dataset report

```bash
python scripts/report_dataset.py \
  --manifests manifests/uwf700.jsonl manifests/fgadr.jsonl \
  --cocos labels_coco/uwf700_train.json labels_coco/uwf700_train_tiles.json \
  --out eval/report_dataset.json
```

Produces:
- `eval/report_dataset.json`

Expect:
- image/class/label-format distribution summary

### Preprocessing report

```bash
python scripts/report_preproc.py \
  --manifest manifests/uwf700.jsonl \
  --preproc-root preproc \
  --out-dir eval/preproc
```

Produces:
- `eval/preproc/preproc_report.json`
- `eval/preproc/raw_vs_norm_grid.png`
- `eval/preproc/roi_hist_before_after.png`

Expect:
- ROI area ratio summary and before/after visual checks

### Training report

```bash
python scripts/report_training.py \
  --run-dir runs/uveitis_ft \
  --out-json eval/training_report.json \
  --out-png eval/training_curves.png
```

Produces:
- `eval/training_report.json`
- `eval/training_curves.png`

Expect:
- epoch curves + best-val summary

### Preprocessing ablation report

```bash
python scripts/ablate_preproc.py \
  --pred-a preds/uveitis_ft \
  --pred-b preds/uveitis_ft_no_norm \
  --out eval/ablate_preproc.json
```

Produces:
- `eval/ablate_preproc.json`

Expect:
- average prediction-count delta across shared images

## 4) Export Artifacts from Host

From host shell after container run:

```bash
tar -czf uveitis_runs.tar.gz runs preds preds_vis eval
```

Produces:
- `uveitis_runs.tar.gz`
