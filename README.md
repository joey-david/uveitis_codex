# Uveitis Codex

Pipeline de localisation/classification de lésions rétiniennes UWF, avec:
- prétraitement ROI via SAM2,
- normalisation couleur ROI-aware,
- entraînement RETFound (mask-first),
- API FastAPI + UI web clinique.

## Démarrage Rapide

### 1) API Docker (serveur GPU)
```bash
docker compose -f deploy/docker-compose.inference-api.yml up -d
docker compose -f deploy/docker-compose.inference-api.yml ps
curl http://127.0.0.1:18080/health
```

### 2) Tunnel SSH (machine locale)
```bash
ssh -N -L 18080:127.0.0.1:18080 joey.david@<SERVER_HOST>
```

### 3) UI locale
```bash
cd webui/clinical-ui
python3 -m http.server 5173
```
Ouvrir `http://127.0.0.1:5173`.

## Documentation (point d'entrée)
- `notes/README.md`: index général des docs.
- `notes/NATIVE_PIPELINE_RUNBOOK.md`: prétraitement, labels, entraînement, évaluation, recalibration.
- `notes/WEB_UI_DEPLOY.md`: déploiement API/UI, tunnel SSH, troubleshooting.
- `webui/clinical-ui/README.md`: usage UI et configuration front.
- `datasets/datasets.md`: structure datasets attendue.

## Gestion Docker
Le repo s'appuie sur l'image `uveitis-codex:latest` (pas de `Dockerfile` versionné ici).

Validation GPU:
```bash
docker run --rm --runtime=nvidia --gpus all uveitis-codex:latest \
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Shell d'entraînement:
```bash
docker run --rm -it \
  --runtime=nvidia --gpus all \
  -v /home/joey.david/uveitis_codex:/workspace \
  -w /workspace \
  uveitis-codex:latest bash
```

## Réentraînement (résumé)
Exécuter depuis `/workspace` (dans le conteneur):
```bash
uv run python scripts/stage0_build_manifest.py --config configs/stage0_manifest.yaml
uv run python scripts/stage0_preprocess.py --config configs/stage0_preprocess.yaml
uv run python scripts/stage0_build_labels.py --config configs/stage0_labels_main9.yaml
uv run python scripts/stage4_adapt_retfound.py --config configs/stage4_adapt_retfound.yaml
uv run python scripts/stage5_train_mask_head.py --config configs/stage5_train_mask_head.yaml
uv run python scripts/stage6_infer_mask_to_obb.py --config configs/stage6_infer_mask_to_obb.yaml
uv run python scripts/stage7_calibrate_thresholds.py --config configs/stage7_calibrate_thresholds.yaml
```

Sorties principales:
- checkpoints/metrics: `runs/retfound_mask/<run_name>/`
- prédictions: `eval/<run_name>/`
- index/labels: `manifests/`, `labels_native*/`
- prétraitement: `preproc*/`

## Références Performances
- checkpoint historique: `checkpoints/main9_ensemble_ap03076/`
- checkpoint overfit UWF: `checkpoints/main9_overfit_ap06739/`

