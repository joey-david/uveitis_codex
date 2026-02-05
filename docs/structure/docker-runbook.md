# Docker Runbook

Use this page as the in-doc reference for running the project in Docker.

## Quick Start

1. Build image:

```bash
docker build -t uveitis-codex:latest .
```

2. Launch GPU container:

```bash
docker run --rm -it --gpus all \
  --shm-size=16g \
  -v "$PWD:/workspace" \
  -w /workspace \
  uveitis-codex:latest bash
```

3. Inside container, run regular pipeline commands (`python scripts/...`) as listed in:
- [`Stage Map`](stage-map.md)

4. Full command-by-command runbook is maintained in the repository root file:
- `docker.md`

Note: `docker.md` is outside the MkDocs docs directory, so it is referenced as a path here (not a docs hyperlink).
