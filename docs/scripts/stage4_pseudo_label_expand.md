# `stage4_pseudo_label_expand.py`

## Purpose
Build an expanded COCO dataset by adding high-confidence pseudo labels from predictions.

## CLI
```bash
python scripts/stage4_pseudo_label_expand.py --base-coco labels_coco/uwf700_train_tiles.json --pred-dir preds/uveitis_ft --min-score 0.9 --out-coco pseudo_labels/uveitis_ft/train_plus_pseudo.json [--unlabeled-coco PATH]
```

## Reads
- Base COCO file.
- Optional unlabeled COCO image set.
- Prediction JSON files.

## Writes
- Expanded COCO file with pseudo annotations.

## Functions
| Function | Description |
|---|---|
| `_load_coco(path)` | Loads a COCO JSON file. |
| `_collect_preds(pred_dir, min_score)` | Collects filtered predictions from a prediction directory. |
| `main()` | Merges base annotations with pseudo labels and saves output COCO. |
