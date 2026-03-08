# Native Pipeline Runbook

Runbook complet pour reconstruire la chaîne entraînement/éval à partir du repo.

## 0) Prérequis

- Image Docker disponible: `uveitis-codex:latest`
- GPU visible depuis Docker (`--runtime=nvidia --gpus all`)
- Datasets en place (`datasets/datasets.md`)
- Poids SAM2/RETFound présents (`models/`)

Vérification rapide GPU:
```bash
docker run --rm --runtime=nvidia --gpus all uveitis-codex:latest \
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Vérification `uv`:
```bash
docker run --rm uveitis-codex:latest uv --version
```

Entrer dans le conteneur de travail:
```bash
docker run --rm -it \
  --runtime=nvidia --gpus all \
  -v /home/joey.david/uveitis_codex:/workspace \
  -w /workspace \
  uveitis-codex:latest bash
```

## 1) Manifests / Splits

```bash
uv run python scripts/stage0_build_manifest.py --config configs/stage0_manifest.yaml
```

État attendu:
- `manifests/*.jsonl`
- `splits/stage0_0.json`

Vérification:
```bash
uv run python scripts/report_dataset.py --manifest manifests/uwf700_labeled.jsonl
```

## 2) Prétraitement (SAM2 ROI -> normalisation ROI -> global/tiles)

```bash
uv run python scripts/stage0_preprocess.py --config configs/stage0_preprocess.yaml
```

État attendu sous `preproc/` (ou dossier configuré):
- `roi_masks/`: masques SAM2 nettoyés (trous fermés via contour externe principal)
- `crops/`: recadrage ROI
- `norm/`: version normalisée ROI-aware
- `global/`: image globale entrée modèle
- `tiles/` + `tiles_meta/`
- `verify/preprocess_metrics.json`

QA prétraitement:
```bash
uv run python scripts/qa_preproc_norm_to_regular.py \
  --config configs/stage0_preprocess.yaml \
  --out eval/preproc_qa
```

## 3) Construction labels natifs (sans COCO)

Classes principales:
```bash
uv run python scripts/stage0_build_labels.py --config configs/stage0_labels_main9.yaml
```

Branche vascularite:
```bash
uv run python scripts/stage0_build_labels.py --config configs/stage0_labels_vascularite.yaml
```

État attendu:
- `labels_native*/**.jsonl`
- `labels_native*/records/**.json`
- `labels_native*/class_map_active.json`

QA labels:
```bash
uv run python scripts/qa_native_labels.py \
  --index labels_native/uwf700_val_global.jsonl \
  --out eval/native_labels_qa_val \
  --n 30
```

## 4) Adaptation RETFound (optionnelle mais recommandée)

```bash
uv run python scripts/stage4_adapt_retfound.py --config configs/stage4_adapt_retfound.yaml
```

Sorties attendues:
- `runs/retfound_adapt/<run_name>/best.pt`
- `runs/retfound_adapt/<run_name>/metrics.json`

## 5) Entraînement mask-first

```bash
uv run python scripts/stage5_train_mask_head.py --config configs/stage5_train_mask_head.yaml
```

Sorties attendues:
- `runs/retfound_mask/<run_name>/best.pt`
- `runs/retfound_mask/<run_name>/metrics.json`

## 6) Inférence (masques -> polygones/OBB/boxes)

```bash
uv run python scripts/stage6_infer_mask_to_obb.py --config configs/stage6_infer_mask_to_obb.yaml
```

Sorties attendues:
- `eval/<run_name>/predictions.jsonl`
- `eval/<run_name>/previews/*.png`

Évaluation native:
```bash
uv run python scripts/eval_native_detection.py \
  --gt labels_native/uwf700_val_global.jsonl \
  --pred eval/<run_name>/predictions.jsonl \
  --out eval/<run_name>/metrics_ap50_v2.json
```

## 7) Calibration des seuils

```bash
uv run python scripts/stage7_calibrate_thresholds.py --config configs/stage7_calibrate_thresholds.yaml
```

Sortie attendue:
- `runs/retfound_mask/<run_name>/calibrated_thresholds.json`

Puis injecter le JSON dans `postprocess.thresholds_json` de `configs/stage6_infer_mask_to_obb.yaml` et relancer stage 6.

Calibration postprocess (optionnelle, utile pour duplicats/overlays):
```bash
uv run python scripts/stage7_calibrate_detection_postprocess.py \
  --pred eval/<run_name>/predictions.jsonl \
  --gt labels_native/uwf700_val_global.jsonl \
  --out runs/retfound_mask/<run_name>/class_postprocess.json
```

## 8) Gestion espace disque (important)

Vérifier volumes:
```bash
du -sh runs eval preproc preproc_main9_fastiter labels_native* checkpoints 2>/dev/null
```

Nettoyage prudent (garder les best checkpoints + rapports):
```bash
find runs -type f -name 'last.pt' -delete
find eval -type f -name '*.tmp' -delete
```

Nettoyage Docker:
```bash
docker system df
docker image prune -f
docker container prune -f
```

## 9) Checkpoints de référence

- `checkpoints/main9_ensemble_ap03076/`
  - meilleure version ensemble historique.
- `checkpoints/main9_overfit_ap06739/`
  - profil overfit UWF interne (fortement in-domain).

## 10) Règles de reproductibilité

- Toujours versionner configs YAML d'expérience dans `configs/experiments/`.
- Toujours sauvegarder métriques + previews dans `eval/<run_name>/`.
- Toujours noter les ablations/écarts dans `notes/` avant cleanup.
