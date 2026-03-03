# Web Inference UI Deployment

This setup has two parts:
- API server (runs on this GPU server, hosts the model)
- Static web UI (can be deployed on any machine/web host)

## 1) Start the API on the GPU server

From repo root:

```bash
uv run python scripts/serve_inference_api.py --config configs/inference_api.yaml --host 0.0.0.0 --port 8080
```

Expected startup behavior:
- model branches are loaded once into GPU memory
- process stays running and serves HTTP endpoints

Quick checks:

```bash
curl http://127.0.0.1:8080/health
```

Expected output:

```json
{"ok":true}
```

```bash
curl http://127.0.0.1:8080/v1/profiles
```

Expected output contains:
- `best_overfit`
- `balanced`

Optional access control:

```bash
export UVEITIS_API_TOKEN='replace-with-strong-token'
export UVEITIS_CORS_ORIGINS='https://your-ui-domain.example,http://localhost:5173'
uv run python scripts/serve_inference_api.py --config configs/inference_api.yaml --host 0.0.0.0 --port 8080
```

When token is set, clients must send:
- `Authorization: Bearer <token>`

### Docker service (recommended)

Compose file:
- `deploy/docker-compose.inference-api.yml`

Start:

```bash
docker compose -f deploy/docker-compose.inference-api.yml up -d
```

Check:

```bash
docker compose -f deploy/docker-compose.inference-api.yml ps
docker compose -f deploy/docker-compose.inference-api.yml logs -f --tail=100
curl http://127.0.0.1:18080/health
```

Stop:

```bash
docker compose -f deploy/docker-compose.inference-api.yml down
```

Optional token + CORS before start:

```bash
export UVEITIS_API_TOKEN='replace-with-strong-token'
export UVEITIS_CORS_ORIGINS='http://127.0.0.1:5173'
docker compose -f deploy/docker-compose.inference-api.yml up -d
```

## 2) Deploy the UI elsewhere

UI folder:
- `webui/clinical-ui`

Edit:
- `webui/clinical-ui/config.js`

Set:
- `apiBaseUrl` to this server API URL (e.g. `https://your-gpu-server.example`)
- `apiToken` if API token is enabled

Serve statically (example):

```bash
cd webui/clinical-ui
python3 -m http.server 5173
```

Open:
- `http://127.0.0.1:5173`

### Reach server API from your local machine

If direct network access to server port `18080` is not available, use an SSH tunnel:

```bash
ssh -N -L 18080:127.0.0.1:18080 joey.david@<SERVER_HOST>
```

Then set in `webui/clinical-ui/config.js`:

```js
window.UVEITIS_UI_CONFIG = {
  apiBaseUrl: "http://127.0.0.1:18080",
  apiToken: ""
};
```

## 3) API contract used by the UI

### `GET /v1/profiles`
- returns available inference profiles

### `POST /v1/predict`
Form fields:
- `file`: image file (`image/*`)
- `profile`: profile name (`best_overfit` by default)

Response includes:
- `predictions` (class, score, bbox, polygon)
- `counts_by_class`
- `timings_ms`
- PNG images as base64:
  - `original_overlay_png_b64`
  - `global_preprocessed_png_b64`
  - `global_overlay_png_b64`
  - `roi_mask_png_b64`

## 4) Notes

- The API profile `best_overfit` is the highest-performing in-domain profile.
- It is intentionally UWF-focused and should be validated externally before clinical use beyond this dataset distribution.
