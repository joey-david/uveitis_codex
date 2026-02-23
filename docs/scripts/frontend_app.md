# `frontend_app.py`

## Purpose
Run the visual frontend UI for 4-step inference review:
1) image selection, 2) SAM ROI view, 3) processing animation, 4) final labels on original image.

## CLI
```bash
streamlit run scripts/frontend_app.py
```

Optional config override:
```bash
UVEITIS_FRONTEND_CONFIG=configs/frontend.yaml streamlit run scripts/frontend_app.py
```

## Reads
- `configs/frontend.yaml`
- local mode: checkpoint + preprocess config through `FrontendInferenceService`
- remote mode: `frontend.remote_api_url` endpoint

## Writes
- local mode writes the same artifact bundle as `frontend_api.py` under `out/frontend/<run_id>/`
- remote mode depends on server settings

## Runtime Modes
- `local`: runs inference in the Streamlit process.
- `remote`: uploads the image to `/infer` on the configured API server.

## Functions
| Function | Description |
|---|---|
| `_inject_css()` | Applies frontend visual style and responsive layout rules. |
| `_get_local_service(config_path)` | Cached local service instance for repeated inference calls. |
| `_run_remote(...)` | Calls remote API and returns JSON payload. |
| `main()` | Builds sidebar controls and renders the 4-step pipeline view. |
