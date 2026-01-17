# Troubleshooting

## Docker sees no GPU
- Confirm host has NVIDIA drivers.
- Use: `docker run --rm --gpus all nvidia/cuda:12.1.1-base nvidia-smi`.

## CUDA OOM
- Reduce --batch-size.
- Lower --image-size (224 -> 160).
- Reduce --num-boxes.

## Manifest errors
- Run `scripts/validate_manifest.py` and fix missing paths/boxes.
- Ensure paths are relative to repo root or pass --root.

## Slow training
- Increase --num-workers if CPU allows.
- Ensure dataset is on local SSD.

## Bad localization
- Verify box format; use --box-format obb if needed.
- Ensure boxes are normalized to image size in manifest inputs (script will normalize after load).
