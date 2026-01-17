# Training (MVCAViT)

## Common flags
- --use-pso: enable PSO fusion tuning
- --mirror-view: duplicate macula view if only one image per record
- --box-format obb: convert OBB labels to axis-aligned boxes

## Stage A: DR pretraining (optional)
```bash
python scripts/train_mvcavit.py \
  --train-manifest manifests/drtid_train.jsonl \
  --val-manifest manifests/drtid_val.jsonl \
  --output-dir runs/mvcavit_dr \
  --num-classes 5 --num-boxes 10 \
  --use-pso
```
Outputs:
- runs/mvcavit_dr/best.pt
- runs/mvcavit_dr/last.pt
- runs/mvcavit_dr/metrics.json

## Stage B: Uveitis fine-tune
```bash
python scripts/train_mvcavit.py \
  --train-manifest manifests/uveitis_train.jsonl \
  --val-manifest manifests/uveitis_val.jsonl \
  --output-dir runs/mvcavit_uveitis \
  --pretrained runs/mvcavit_dr/best.pt \
  --mirror-view \
  --box-format obb
```

## Evaluation
```bash
python scripts/eval_mvcavit.py \
  --manifest manifests/uveitis_val.jsonl \
  --checkpoint runs/mvcavit_uveitis/best.pt \
  --mirror-view \
  --box-format obb
```
Expected: JSON with accuracy and mean IoU.
