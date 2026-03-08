# Déploiement API + UI Clinique

Guide complet pour exposer le modèle du serveur GPU vers une UI locale ou distante.

## 1) API d'inférence sur le serveur GPU

L'API est servie par `scripts/serve_inference_api.py` avec config `configs/inference_api.yaml`.

### Option recommandée: Docker Compose

Fichier:
- `deploy/docker-compose.inference-api.yml`

Démarrage:
```bash
docker compose -f deploy/docker-compose.inference-api.yml up -d
```

Vérification:
```bash
docker compose -f deploy/docker-compose.inference-api.yml ps
docker compose -f deploy/docker-compose.inference-api.yml logs -f --tail=120
curl http://127.0.0.1:18080/health
curl http://127.0.0.1:18080/v1/profiles
```

Arrêt / redémarrage:
```bash
docker compose -f deploy/docker-compose.inference-api.yml down
docker compose -f deploy/docker-compose.inference-api.yml up -d
```

### Option directe (sans compose)
```bash
uv run python scripts/serve_inference_api.py \
  --config configs/inference_api.yaml \
  --host 0.0.0.0 --port 8080
```

## 2) Sécurité API (token + CORS)

Variables supportées:
- `UVEITIS_API_TOKEN`: active auth `Bearer <token>`
- `UVEITIS_CORS_ORIGINS`: liste CSV des origines front autorisées

Exemple:
```bash
export UVEITIS_API_TOKEN='replace-with-strong-token'
export UVEITIS_CORS_ORIGINS='http://127.0.0.1:5173,https://votre-ui.example'
docker compose -f deploy/docker-compose.inference-api.yml up -d
```

## 3) Accès depuis la machine locale (SSH tunnel)

Depuis le PC local:
```bash
ssh -N -L 18080:127.0.0.1:18080 joey.david@<SERVER_HOST>
```

Ensuite, côté local:
```bash
curl http://127.0.0.1:18080/health
```
Doit retourner `{"ok":true}`.

## 4) Déploiement UI

Dossier UI:
- `webui/clinical-ui`

Configurer l'endpoint API:
- `webui/clinical-ui/config.js`

Exemple local tunnelé:
```js
window.UVEITIS_UI_CONFIG = {
  apiBaseUrl: "http://127.0.0.1:18080",
  apiToken: ""
};
```

Servir la UI en local:
```bash
cd webui/clinical-ui
python3 -m http.server 5173
```

Ouvrir:
- `http://127.0.0.1:5173`

## 5) Fonctionnalités UI actuellement en place

- interface française,
- état API (en ligne/hors ligne),
- barre de progression globale,
- étapes pipeline (connexion, transmission, prétraitement, inférence, rendu),
- télémétrie transfert/temps (octets, débit, temps serveur, RTT),
- tableaux/classes détectées,
- zoom plein écran sur images résultat.

## 6) Contrat API utilisé par l'UI

### `GET /health`
Retour:
```json
{"ok": true}
```

### `GET /v1/profiles`
Retourne les profils disponibles (`best_overfit`, `balanced`, etc.).

### `POST /v1/predict`
Form-data:
- `file`: image (`image/*`)
- `profile`: nom du profil (par défaut `best_overfit`)

Retour:
- `predictions` (classe, score, bbox, polygone/obb)
- `counts_by_class`
- `timings_ms`
- `images.*_png_b64` pour affichage front

## 7) Troubleshooting

- `304` dans les logs UI: normal (cache navigateur).
- `404 /favicon.ico`: corrigé par `webui/clinical-ui/favicon.svg`; si persiste, hard refresh (`Ctrl+Shift+R`).
- `API offline` dans UI:
  - vérifier `docker compose ... ps`
  - vérifier `curl http://127.0.0.1:18080/health` sur serveur
  - vérifier tunnel SSH actif côté local
  - vérifier `apiBaseUrl` dans `config.js`
- `401 Unauthorized`:
  - token activé côté API mais absent/incorrect côté UI.
- latence élevée:
  - vérifier charge GPU (`nvidia-smi`) et saturation I/O.

## 8) Emplacements utiles

- Compose: `deploy/docker-compose.inference-api.yml`
- API server: `scripts/serve_inference_api.py`
- Service inference: `src/uveitis_pipeline/inference_service.py`
- Config API/profils: `configs/inference_api.yaml`
- UI clinique: `webui/clinical-ui/`
