# `stage0_build_labels.py`

## Purpose
Convert source annotations (UWF OBB / FGADR masks) into COCO labels for global and tile spaces.

## CLI
```bash
python scripts/stage0_build_labels.py --config configs/stage0_labels.yaml
```

## Reads
- Input manifests
- Split JSON
- Class map YAML
- Preprocessing metadata (`preproc/crop_meta`, `preproc/tiles_meta`)

## Writes
- `labels_coco/<dataset>_<split>.json`
- `labels_coco/<dataset>_<split>_tiles.json`
- `labels_coco/summary.json`
- `labels_debug/<dataset>_<split>/*.png`

## Functions
| Function | Description |
|---|---|
| `main()` | Loads manifests/splits/class map, builds COCO labels for selected datasets/splits, saves summary. |

## Core module dependencies
- [`uveitis_pipeline.labels`](../api/labels.md)
- [`uveitis_pipeline.common`](../api/common.md)
