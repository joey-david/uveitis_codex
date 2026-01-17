# Setup (server)

Assumptions:
- CUDA GPUs available in Docker (A100 40GB).
- Repo checked out on server.

## 1) Build Docker image
```bash
docker build -t uveitis-codex:latest .
```
Expected: build logs end with successful image tag.

## 2) Run container (GPU)
```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" -w /workspace \
  uveitis-codex:latest bash
```
Expected: shell prompt inside container at `/workspace`.

## 3) Install deps (only if not baked into image)
```bash
pip install -r requirements.txt
```
Expected: no errors.

## 4) Quick smoke test (CPU only)
```bash
pytest -q
```
Expected: `3 passed`.
