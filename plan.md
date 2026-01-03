## Updated definitive plan (RETFound ViT + UWF-700 + 3 datasets)

**Datasets you will use (fixed):**

1. **UWF-700** (given)
2. **DeepDRiD** (use both its regular fundus + UWF subset)
3. **FGADR** (lesion segmentation masks)
4. **EyePACS** (DR grading, classification-only at scale)

* **your 100 UWF uveitis images** (OBB labels)

**Core tooling (fixed):**

* **Backbone:** RETFound ViT (encoder)
* **Detector:** **MMRotate Oriented R-CNN** (angle convention: **le90**)
* **Rotated box regression:** **GWD loss** (robust to lax/noisy boxes)
* **Training/inference:** tile-based UWF pipeline + merge

---

## Stage 1 — Make RETFound “UWF-native” (self-supervised)

**Initialize encoder:** RETFound ViT weights.
**Train objective:** MAE continuation (same style as RETFound).
**Training images (unlabeled):**
* UWF-700 (all)
* DeepDRiD **UWF subset** (all)
* your uveitis UWF images (ignore labels, use as unlabeled too)

**Metric:** **reconstruction MSE loss** (monitor convergence; target < baseline on natural images).

---

## Stage 2 — Lesion morphology pretraining (segmentation on FGADR)

**Model:** UWF-adapted RETFound encoder + lightweight segmentation decoder/head.
**Dataset:** **FGADR segmentation masks**.
**Task:** multi-class lesion segmentation (FGADR lesion taxonomy).

Train encoder with a lower LR than the decoder (but you do update it).

**Metric:** **Dice Score > 0.70** (mean across lesion classes on validation split).

---

## Stage 3 — Convert FGADR masks → rotated boxes (OBB pretraining set)

For each FGADR lesion class:
1. connected components per mask
2. each component → rotated min-area rectangle → OBB (le90)
3. filter tiny/noise components

**Metric:** **Box Count Distribution** (verify # of lesions per image matches segmentation statistics).

---

## Stage 4 — Pretrain the final detector on FGADR-derived OBBs

**Model:** MMRotate **Oriented R-CNN**
**Backbone init:** encoder from Stage 2
**Train on:** FGADR-derived OBB detection set (Stage 3)
**Loss:** include **GWD** for rotated box regression

**Metric:** **Rotated mAP50 > 30%** (on FGADR val set; this confirms the detector learns lesion concepts).

---

## Stage 5 — Train a classifier teacher (EyePACS + DeepDRiD) for distillation

**Teacher model:** RETFound-based classifier (separate from the detector).

1. **Train on EyePACS** for DR grading (big scale signal).
2. **Fine-tune on DeepDRiD regular fundus + then DeepDRiD UWF** (so the teacher doesn’t collapse on UWF appearance).

**Metric:** **AUC > 0.85** or **Quadratic Weighted Kappa > 0.80** (on DeepDRiD UWF validation).

---

## Stage 6 — Final uveitis training (your real target)

**Student model:** Oriented R-CNN from Stage 4.
**Train on:** your 100 uveitis UWF images (tile-based) with your OBB labels.
**Key choices (fixed):**
* **GWD** for box regression (handles “region boxes” better than plain Smooth-L1)
* class-balanced sampling / reweighting (your long tail is extreme)

**Add distillation from Stage 5 teacher during this training:**
* For each full image, aggregate the detector’s tile predictions into an image-level “abnormality / DR-like lesion” score.
* Add a loss to match the teacher’s probabilities (regularizes features and reduces false positives).

**Metric:** **Rotated mAP50** (primary) + **Sensitivity @ 1.0 FP/image** (clinical relevance).

---

## Hyperparameters & Guidelines (for 4000x3000px Input)

Given the high resolution (**4000x3000px**), you **must** use a tiling approach.

### 1. Tiling Strategy
*   **Tile Size:** `1024x1024` pixels.
*   **Stride:** `768` pixels (25% overlap).
*   **Tiles per Image:** Approx `5x4 = 20` tiles per image (covering 4000x3000).

### 2. Model Input Resolution
RETFound ViT-L is computionally heavy. You should **resize tiles** before feeding to the network.
*   **Training Input:** Resize `1024x1024` tile → **`512x512`** (or `224x224` if OOM).
    *   *Note:* `512x512` captures more fine-grained lesion detail (MA/microaneurysms) than `224`.
    *   You will need to interpolate RETFound's positional embeddings from 224 to 512 (standard ViT practice).

### 3. Detector Anchors & Scales
*   **Anchor Scales:** Since we resize `1024 -> 512` (0.5x scale), ensure anchor sizes cover the *effective* lesion sizes.
*   **Smallest Lesions:** A 10px microaneurysm becomes 5px. Ensure the RPN anchor generator includes small scales (e.g., scale 4 or 8).

### 4. Training Config (Guidelines)
*   **Batch Size:** 2-4 (per GPU) for ViT-L @ 512px. Use **Gradient Accumulation** to effectively reach batch 16+.
*   **Learning Rate:** ViT requires lower LR than ResNet. Start `1e-4` (AdamW) with `0.05` weight decay. Layer-wise LR decay (lower LR for earlier backbone layers) is highly recommended for transfer learning.
*   **Epochs:**
    *   Stage 1 (MAE): 400-800 epochs (needs long training).
    *   Stage 4/6 (Detection): 12-24 epochs (1x or 2x schedule) is usually sufficient if pretraining was good.
