## Docker workflow (remote 40GB A100)

This project is designed to run entirely inside Docker on a remote GPU host.

## 1) Build image

```bash
docker build -t uveitis-codex:latest .
```

Expected output:
- normal Docker build logs
- final line similar to `Successfully tagged uveitis-codex:latest`

## 2) Start training container with GPU

```bash
docker run --rm -it --gpus all \
  --shm-size=16g \
  -v "$PWD:/workspace" \
  -w /workspace \
  uveitis-codex:latest bash
```

Expected output:
- interactive shell prompt in `/workspace`

Alternative:

```bash
docker compose run --rm train bash
```

## 3) Verify GPU visibility in container

```bash
python - <<'PY'
import torch
print({'cuda_available': torch.cuda.is_available(), 'gpu_count': torch.cuda.device_count()})
PY
```

Expected output:
- `{'cuda_available': True, 'gpu_count': 1}` (or more GPUs)

## 4) Stage 0.0: build manifests and split

```bash
python scripts/stage0_build_manifest.py --config configs/stage0_manifest.yaml
```

Expected stdout:
- `Wrote manifests to manifests`
- `Wrote split file to splits/stage0_0.json`
- dataset counts per split/label type/class

Artifacts created:
- `manifests/*.jsonl`, `manifests/*.csv`
- `splits/stage0_0.json`

## 5) Stage 0.1-0.3: preprocess images

```bash
python scripts/stage0_preprocess.py --config configs/stage0_preprocess.yaml
```

Expected stdout:
- `Preprocessing complete`
- metric dict (ROI fail rates, tile distribution, reconstruction error)

Artifacts created:
- `preproc/roi_masks/*.png`
- `preproc/crops/*.png`
- `preproc/crop_meta/*.json`
- `preproc/norm/*.png`
- `preproc/norm_meta/*.json`
- `preproc/global_1024/*.png`
- `preproc/tiles/{image_id}/*.png`
- `preproc/tiles_meta/*.json`
- `preproc/verify/*.png`, `preproc/verify/preprocess_metrics.json`

## 6) Stage 0.4: build COCO labels

```bash
python scripts/stage0_build_labels.py --config configs/stage0_labels.yaml
```

Expected stdout:
- summary lines per dataset/split with `num_images`, `num_annotations`, class counts

Artifacts created:
- `labels_coco/{dataset}_{split}.json`
- `labels_coco/{dataset}_{split}_tiles.json`
- `labels_coco/summary.json`
- `labels_debug/{dataset}_{split}/*.png`

## 7) Stage 1.0: overfit sanity run (10 images)

```bash
python scripts/train_detector.py --config configs/train_overfit10.yaml
```

Expected output behavior:
- training starts immediately
- checkpoint files appear under `runs/overfit_10/checkpoints`

Artifacts created:
- `runs/overfit_10/config.yaml`
- `runs/overfit_10/checkpoints/epoch_*.pth`
- `runs/overfit_10/checkpoints/best.pth`
- `runs/overfit_10/checkpoints/last.pth`
- `runs/overfit_10/metrics.jsonl`
- `runs/overfit_10/val_report.json`

## 8) Stage 2.1: FGADR supervised pretrain

```bash
python scripts/train_detector.py --config configs/train_fgadr.yaml
```

Artifacts:
- `runs/fgadr_pretrain/...`

## 9) Stage 3.1: Uveitis fine-tune

```bash
python scripts/train_detector.py --config configs/train_uveitis_ft.yaml
```

Artifacts:
- `runs/uveitis_ft/...`
- `runs/uveitis_ft/val_report.json` (includes per-class sensitivity at FP/image)

## 10) Stage 1.1 inference merge (tiles -> global)

```bash
python scripts/infer_detector.py --config configs/infer_uveitis_ft.yaml
```

Artifacts:
- `preds/uveitis_ft/{image_id}.json`
- `preds_vis/uveitis_ft/{image_id}.png`

## 11) Reports and bottleneck hooks

```bash
python scripts/report_dataset.py --manifests manifests/uwf700.jsonl manifests/fgadr.jsonl --cocos labels_coco/uwf700_train.json labels_coco/uwf700_train_tiles.json --out eval/report_dataset.json
python scripts/report_preproc.py --manifest manifests/uwf700.jsonl --preproc-root preproc --out-dir eval/preproc
python scripts/report_training.py --run-dir runs/uveitis_ft --out-json eval/training_report.json --out-png eval/training_curves.png
python scripts/ablate_preproc.py --pred-a preds/uveitis_ft --pred-b preds/uveitis_ft_no_norm --out eval/ablate_preproc.json
```

## 12) Optional stage 4 scripts

Pseudo-label expansion:

```bash
python scripts/stage4_pseudo_label_expand.py \
  --base-coco labels_coco/uwf700_train_tiles.json \
  --pred-dir preds/uveitis_ft \
  --out-coco pseudo_labels/uveitis_ft/train_plus_pseudo.json
```

RetFound MAE continuation hook (only runs if `../RETFound/main_pretrain.py` exists):

```bash
python scripts/stage4_continue_mae.py --retfound-dir ../RETFound --data-path preproc/global_1024 --run
```

## 13) Copy artifacts from remote host

If needed from host machine:

```bash
tar -czf uveitis_runs.tar.gz runs preds preds_vis eval
```

Expected output:
- `uveitis_runs.tar.gz` in project root
