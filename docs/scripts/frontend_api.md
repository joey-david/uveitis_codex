# `frontend_api.py`

## Purpose
Run an HTTP inference server for the frontend (single-image SAM ROI + detector labels).

## CLI
```bash
python scripts/frontend_api.py --config configs/frontend.yaml --host 0.0.0.0 --port 8000
```

## Reads
- `configs/frontend.yaml`
- `configs/infer_*.yaml` (via `pipeline.infer_config`)
- `configs/stage0_preprocess.yaml` (via `pipeline.preprocess_config`)
- model checkpoint path from inference config
- SAM/SAM2 checkpoints if configured and available

## Writes
- `out/frontend/<run_id>/input.png`
- `out/frontend/<run_id>/roi_mask.png`
- `out/frontend/<run_id>/roi_overlay.png`
- `out/frontend/<run_id>/masked_raw.png`
- `out/frontend/<run_id>/normalized_crop.png`
- `out/frontend/<run_id>/detector_input.png`
- `out/frontend/<run_id>/final_overlay.png`
- `out/frontend/<run_id>/juxtaposed.png`
- `out/frontend/<run_id>/result.json`

## API
- `GET /health`
  - Returns status, device, checkpoint, and default runtime values.
- `POST /infer`
  - Multipart upload field: `file`
  - Optional form fields: `score_thresh`, `dataset`
  - Returns predictions + base64 PNGs for all intermediate/final views.

## Functions
| Function | Description |
|---|---|
| `build_app(config_path)` | Creates FastAPI app with `/health` and `/infer`. |
| `main()` | Parses CLI args and starts Uvicorn. |
