Build image:
```bash
docker build -t uveitis-codex:latest .
```

Run GPU container for training:
```bash
docker run --rm -it --gpus all -v "$PWD:/workspace" -w /workspace uveitis-codex:latest bash
```

Run downloader:
```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  uveitis-codex:latest \
  bash -lc "python datasets/drive_dl.py"
```

Verify zip:
```bash
ls -lh datasets_bundle.zip
```
