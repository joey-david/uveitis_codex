# Self-FI Paper: What To Copy Here

Paper in repo: `nature_vit_cnn_dr_localization.pdf` (Nguyen et al., Bioengineering 2023).

## What They Actually Do
- They propose a self-supervised contrastive pretraining setup for fundus **classification** (not detection).
- Core idea: create additional *positive pairs* beyond SimCLR’s “two augmentations of the same image”.
- Three losses (their Eq. 4):
  - `Lpair-instance`: classic SimCLR-style two augmentations of the same image.
  - `Lbi-lateral`: left vs right eye images from the same patient.
  - `Lmulti-modality`: UWF (UFI) vs conventional fundus (CFI) images from the same patient.

## Constraints In Our Repo
- We don’t reliably have patient identity and “left/right” pairing for uveitis images.
- We also don’t have true UWF<->CFI pairs from the *same patient*.

## Practical Adaptation For Our Pipeline
We can still borrow the *multi-positive-pair* idea by constructing “two modalities” from the same source image:
- **UWF modality A**: masked fundus-only global (`preproc/global_1024/...png`).
- **UWF modality B** (proxy for “CFI modality”):
  - the same image after our “normalize-to-regular-fundus” color stats, or
  - a center-crop / ROI crop (more similar to conventional fundus FOV), or
  - a different photometric pipeline (CLAHE, gamma) applied *after masking*.

This gives a Self-FI-like `Lmulti-modality` without needing same-patient cross-modal pairing.

## How This Helps Detection (Not Just Classification)
Two integration points that are cheap and measurable:
1. **Image-level class prior to reduce FP**:
   - Train a multi-label classifier (Main9 classes only) on globals using image-level labels aggregated from COCO tiles.
   - At inference, suppress/scale YOLO OBB scores for classes the classifier says are absent.
2. **Box-level re-scoring to reduce duplicates**:
   - Crop around each predicted OBB, re-score with the classifier.
   - This is a second-stage filter that specifically targets the “many overlapping same-class boxes” failure mode.

## Immediate Fix We Implemented (Orthogonal, But Needed)
- Our duplicate OBB stacks are largely caused by *overlapping tiles*.
- We now run a **global polygon NMS across all tiles** when generating previews, so duplicates collapse cleanly.

