## MVCAViT Plan (CNN + ViT + PSO)

**Goal:** Multi-view classification + lesion localization with MVCAViT, adapted for uveitis bounding boxes.

### Stage 1 — Data prep
- Build paired-view manifests (macula/optic) or mirror single-view uveitis images.
- Validate manifests (file existence, box bounds).

### Stage 2 — MVCAViT training
- Train MVCAViT with cross-attention fusion and multitask heads.
- Optional PSO to optimize fusion weights.
- Output checkpoints and metrics in `runs/mvcavit/`.

### Stage 3 — Transfer to uveitis
- Initialize from DR-trained checkpoint.
- Fine-tune on uveitis boxes with `--mirror-view` if single-view.

### Stage 4 — Evaluation and iteration
- Track accuracy and IoU via `scripts/eval_mvcavit.py`.
- Inspect errors in manifests and box stats for bottleneck debugging.
