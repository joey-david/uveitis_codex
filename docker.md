Build image:
```bash
docker build -t uveitis-codex:latest .
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
