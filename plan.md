I need to develop a model or pipeline that localizes and classifies symptoms typical of uveitis on Ultra Wide Field (UWF) fundus images. 

Pros:
- Despite Uveitis being a rare disease for which very few accessible datasets exist, most if not all of its fundus symptoms are shared with Diabetic Retinopathy, which is much more widespread, including in online datasets.
- I have a 40GB A100 at my disposal.
- Since Uveitis is a rare disease, our objectives are pretty modest - we can focus on detecting the 6-7 most common symptoms instead of absolutely all of them, and our accuracy and recall should be high but don't have to be perfect.
- There exists a ViT trained on retina images called Retfound whose internal representations should be very useful to our end.
- We have at our disposal a dataset of UWF uveitis funduses labeled with traditional uveitis symptoms with OBBs + 1 label of symptom type per OBB.

Cons:

- Most localization+classification datasets for diabetic retinopathy are on regular fundus images instead of UWF. There are significant differences between the two, and UWF images are less consistent than regular fundus images depending on the dataset. Since they are so wide, one can often see more than the actual fundus on them - eyelashes and part of the eyelid, part of the machine, etc. They also aren't of a consistent circular shape, as the "whole" fundus is captured in them - boundaries of what to consider and what to ignore are less obvious. The color varies a lot and generally isn't consistent with regular fundus images (at least not without preprocessing). Attached is an UWF images (the one with the green tint) and a regular fundus image (the one with the orange tint).
- Most DR datasets focus on grading severity of the disease rather than actually localizing defects.
- Retfound was trained on regular fundus images.
- our dataset consists only of 100 images, and the OBBs are pretty lax - they're not strict masks, and are more indications of where the symptom is. The ophtalmologist who created it just circled the symptoms and wrote a number next to each circle, which I've converted to OBBs programmatically.

To train our model, we will further be able to rely (we don't have to use all of them) the following datasets:
- UWF-700: 700 UWF images with 100 images per disease type. For the 100 corresponding to Uveitis, we have labels (obbs+classification of symptom) created by the ophtalmologist.
- FGADR: see attached paper. Regular fundus images, but with extensive masks classified per symptom. 
- eyepacs: A huge (~50k) bank of medium resolution (~60kB per image) regular fundus images, classified by DR severity (0 through 4).
- deepdrid, containing both regular and UWF images, graded by severity.


Core decisions (opt for trustworthiness + speed)

1. **Start with HBB (axis-aligned boxes), even if you have OBB.** Your OBBs came from circles → the angle is mostly label noise. HBB is more robust, plugs directly into mature tooling, and lets you reuse FGADR (masks→boxes) without friction.
   *If later you find one class truly directional (e.g., sheathing along vessels), add a rotated head only for that class as a v2.*

2. **Use a two-stage detector (Faster R-CNN or Cascade R-CNN), not YOLO-first.** Two-stage is typically more forgiving with loose boxes and small datasets.

3. **Leverage RetFound as the backbone, but don’t make “full MAE re-pretraining” a hard dependency.** It’s great if your RetFound codebase supports it easily; otherwise you’ll get 80% of the benefit via (i) UWF normalization + (ii) supervised transfer on FGADR.

---

## Final pipeline

### Step 0 — Preprocessing that makes UWF behave

You want a deterministic preprocessing script you can run on every dataset.

**0.1 Retina ROI mask (fast, good-enough)**

* Start with a classical approach: downsample → threshold (on intensity or saturation) → largest connected component → fill holes → dilate/erode.
* Crop to the ROI bounding box, **pad to square**, resize.

If this fails on too many images, train a tiny U-Net for “retina-valid pixels”, but only after you’ve tried the heuristic (most projects don’t need the U-Net).

**0.2 Retina-only photometric normalization (avoid overengineering)**

* Compute mean/std per channel **inside the ROI mask** and do per-image z-score (or simple “gray-world” color constancy).
* Optionally add CLAHE **inside the ROI**.
  This reduces the “green UWF vs orange fundus” gap without trying to turn UWF into fake RGB fundus.

**0.3 Resolution + tiling**

* Keep a “global” resized view (e.g., 1024×1024) for context.
* Train/infer with **tiles** (e.g., 512 or 768 with overlap) so micro-lesions survive. Store the mapping back to global coordinates.

This preprocessing is non-negotiable: it’s the cheapest way to make everything downstream reliable.

---

### Step 1 — Model: RetFound-ViTDet detector

**Backbone:** RetFound ViT (start with whatever size is easiest; ViT-B is often enough and faster to iterate).
**Neck:** SimpleFPN (multi-scale features from ViT).
**Head:** Faster R-CNN (or Cascade R-CNN if you already have it working).

Why this combo:

* ViTDet + SimpleFPN solves the “ViT is single-scale” issue.
* Two-stage detection handles noisy localization better.
* Everything is standard in Detectron2-style frameworks, so you’re not debugging research code.

---

### Step 2 — Training curriculum (minimum viable + optional boost)

#### Stage A (recommended): Supervised lesion pretraining on FGADR

Goal: teach the network “lesion primitives” with dense labels.

* Convert FGADR masks → HBB boxes (one box per connected component per class).
* Train the detector on FGADR boxes.
* Use strong *photometric* aug (brightness/contrast, mild color jitter, blur), but avoid aggressive geometric warps that distort anatomy.

This stage is where you get real localization skill cheaply.

#### Stage B (target): Fine-tune on your UWF uveitis OBBs → converted to HBB

Goal: adapt to UWF look + your symptom taxonomy.

**Make lax boxes workable (one high-impact tweak):**

* Lower RPN/ROI positive IoU thresholds (e.g., pos at 0.4–0.5 instead of 0.7).
* Increase the number of proposals / keep more candidates (recall > precision early).
* Use class-balanced sampling or reweighting (uveitis symptoms will be imbalanced).

Also:

* Freeze early ViT blocks for the first part of fine-tuning, then unfreeze gradually.
* Keep LR low, use early stopping, and do k-fold CV if possible.
* Freeze the early layers of the RetFOUND vit.

#### Stage A′ (optional, only if easy): Self-supervised UWF adaptation of RetFound

If your RetFound repo already supports “continue MAE on new images” in a day:

* Run MAE continuation on unlabeled UWF (UWF-700 non-uveitis + DeepDRiD UWF) **after preprocessing**.
  Then proceed to Stage A and B.

If it’s not plug-and-play, skip it. The supervised FGADR→uveitis transfer usually buys you more per unit engineering time.

#### Stage C (optional): pseudo-label expansion on UWF

Once Stage B works:

* Run inference on unlabeled UWF, keep only high-confidence detections, retrain a few epochs.
  This can help, but only do it after you have a stable baseline.

---

## Inference (how you get clinically usable outputs)

* Run on overlapping tiles.
* Merge predictions back to global coordinates.
* Apply NMS (class-wise), then produce:

  * final boxes + symptom labels + confidence,
  * optionally a “lesionness heatmap” by accumulating box scores (cheap and interpretable).

---

## What you’ll get quickly (and what to postpone)

**You can get a credible detector fast** with: preprocessing + FGADR pretrain + uveitis fine-tune + tiling inference.

---

## Checklist to keep it robust

* Verify ROI masking on ~50 random UWF images before training.
* Ensure the label taxonomy is consistent across datasets (map FGADR lesion types → your symptom classes where appropriate; treat the rest as “uveitis-only” classes learned purely in Stage B).
* Evaluate with relaxed IoU (because labels are lax) *and* per-class sensitivity at fixed FP/image (more meaningful clinically).
