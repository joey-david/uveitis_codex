# Uveitis Codex Documentation

This documentation covers the pipeline scaffold implemented in this repository:
- folder and artifact contracts
- stage/program entrypoints
- YAML configuration schema
- module + function-level API reference

Start here:
1. [Folder Reference](structure/folder-reference.md)
2. [Stage Map](structure/stage-map.md)
3. [Docker Runbook](structure/docker-runbook.md)
4. [Script Catalog](scripts/index.md)
5. [Config Catalog](configs/index.md)
6. [API Reference](api/index.md)

How to use this as a newcomer:
1. Read [Stage Map](structure/stage-map.md) for the full flow.
2. Run one stage script from [Script Catalog](scripts/index.md).
3. Open the matching YAML in [Config Catalog](configs/index.md).
4. If you need implementation detail, jump to [API Reference](api/index.md).

Optional rendered docs site (MkDocs):
```bash
pip install -r requirements-docs.txt
mkdocs serve
```
Then open `http://127.0.0.1:8000`.
