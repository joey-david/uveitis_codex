IMPORTANT: WHILE THE DEVELOPMENT OCCURS LOCALLY, THE EVENTUAL PROCESSING/ETC WILL NEED TO BE RUN ON A REMOTE SERVER (SEE AGENTS.MD). ALL THAT IS DEVELOPED HERE SHOULD BE CONCISELY BUT WELL-DOCUMENTED, AND EASILY REPRODUCIBLE ON THE REMOTE SERVER. The datasets will be redownloaded and processed there, with only minimal dev local downloads for testing and well function assertion.
Throughout the implementation, you should develop tests and benchmarks to assess the performance of each of the intermediate and final models after their training/adaptation.

## 0) Standardize everything into one geometry + data format

**Tooling:** OpenCV + MMRotate data format

1. **Convert your label format (`cls xyxyxyxy` normalized polygon)** → rotated box `(cx, cy, w, h, θ)` in pixels using `cv2.minAreaRect`, and store in an MMRotate-supported angle convention (use **le90** everywhere).
2. Implement a **UWF tiling loader** (train + test):

   * fixed tile size (e.g., 1024–1536 px), overlapping stride
   * map tile coords ↔ full-image coords for OBBs
3. Fix a **patient-level split** and never change it.

(Everything after this assumes “OBB = le90” and “training = tiles, inference = tiles + merge”.)

---

## 1) Choose the backbone: RETFound ViT-L as the one backbone for everything

**Backbone:** **RETFound ViT-L/16 MAE weights** (open weights + code). ([GitHub][1])

Why: it’s retina-specific and label-efficient, which is exactly your regime.

---

## 2) UWF domain adaptation (self-supervised MAE continuation) on real UWF images

**Datasets (UWF, open):**

* **UWF-700** (700 high-res UWF images; open access via figshare/Sci Data). ([Nature][2])
* **DeepDRiD UWF subset (256 UWF images)**. ([ScienceDirect][3])
* **Your 100 UWF uveitis images** (use them as unlabeled here too).

**Action:** continue **MAE pretraining** starting from RETFound weights (same architecture) on the union of these UWF images.

Goal: make the encoder “UWF-native” before any supervised task.

---

## 3) Lesion segmentation pretraining (dense supervision) for transferable lesion morphology

**Framework:** train a segmentation model whose **encoder is RETFound (from Step 2)**.

**Segmentation datasets (open):**

* **FGADR Seg-set** (1842 images, fine-grained lesion masks incl. hemorrhage/exudates + advanced lesions). ([CSYizhou][4])
* **DDR lesion subset** (pixel-level lesions available; DDR also provides broader DR resources). ([GitHub][5])
* **IDRiD** (ISBI 2018 lesion segmentation: hemorrhages, hard/soft exudates, microaneurysms). ([Idrid][6])
* **E-ophtha EX** (exudate masks). ([ADCIS][7])
* **DiaRetDB1** (lesion masks; small but helpful diversity). ([Kaggle][8])

**Task definition:** multi-class lesion segmentation for the DR lesion vocabulary (at minimum: hemorrhage, hard exudate, soft exudate, microaneurysm; plus FGADR’s fine-grained lesions if you include them).

**Output of Step 3:** a RETFound encoder that is explicitly trained to represent **lesion boundaries and texture**, not just image-level disease.

---

## 4) Convert those segmentation masks into a large rotated-box pretraining set

For every segmentation dataset image:

1. connected components per lesion class
2. each component → `minAreaRect` → rotated box
3. keep boxes above a min area threshold (remove tiny specks/noise)

This produces **tens of thousands of pseudo-OBBs** (especially from FGADR + DDR), aligned with your final detection objective.

---

## 5) Train a DR classifier teacher (separate model) to inject classification-only signal cleanly

This is where “classification-only DR datasets” become useful without corrupting your localization-pretrained backbone.

**Teacher model:** RETFound encoder (start from Step 2 weights) + classification head.

**Classification datasets (open / widely used):**

* **EyePACS / Kaggle Diabetic Retinopathy Detection** (image-level DR grades). ([Kaggle][9])
* **DDR grading** (13,673 images, DR severity). ([Kaggle][10])
* **APTOS 2019** (3,662 images, DR grades). ([Academic Torrents][11])
* Add **DeepDRiD regular + UWF labels** for explicit UWF grading signal. ([ScienceDirect][3])
* Add **ODIR-5K** for multi-disease retinal semantics (helps robustness beyond DR). ([odir2019.grand-challenge.org][12])

**Output of Step 5:** a strong frozen classifier teacher.

---

## 6) Final model: Oriented R-CNN in MMRotate, initialized from lesion-seg RETFound

**Detector framework:** **MMRotate** with **Oriented R-CNN**. ([GitHub][13])

**Why this specific detector:** two-stage rotated detection tends to be more data-efficient and stable for small/rare objects than one-stage OBB YOLO.

### 6.1 Pretrain detector on pseudo-OBBs (Step 4)

* Backbone init: **RETFound encoder from Step 3** (lesion-seg pretrained).
* Train Oriented R-CNN on the large pseudo-OBB set first.

### 6.2 Fine-tune detector on your uveitis OBB dataset (final target)

**Core choices (decided):**

* **Use GWD loss** for rotated box regression to tolerate lax/noisy boxes and still learn when overlaps are poor. ([Proceedings of Machine Learning Research][14])
* **Angle convention:** le90 throughout (consistent training/inference).
* **Training/inference:** tile-based + merge.

### 6.3 Add classifier-teacher distillation during uveitis fine-tune (this is the “classification-only transfer”)

During uveitis detector training, compute an **image-level presence score** per symptom by aggregating box confidences over tiles (e.g., `1 - Π(1 - p_i)`), and add a loss to match the **teacher’s image-level probabilities** for related categories (DR lesions and/or broader disease labels).

This is how you exploit EyePACS/APTOS/DDR classifiers without needing any localization labels from them.

---

## 7) Vessel prior for vasculitis (baked in, not optional)

For **vascularite**, you want vessel-context.

**Train a vessel segmentation model** and run it as preprocessing to produce a vessel probability map channel.

* **DRIVE** (classic vessel segmentation). ([drive.grand-challenge.org][15])
* **PRIME-FP20** (UWF vessel segmentation; small but UWF-specific). ([University Lab Sites][16])

Then feed `(RGB + vessel_map)` into the detector (4-channel input) during uveitis fine-tune and inference.

---

## 8) Deliverable model and evaluation

**Final deliverable:** MMRotate Oriented R-CNN (RETFound backbone), trained:

1. UWF MAE adaptation →
2. lesion segmentation →
3. pseudo-OBB pretrain →
4. uveitis OBB fine-tune (GWD) + classifier distillation + vessel-map channel

**Report:** rotated mAP plus clinically useful sensitivity at fixed FP/image, on the fixed patient split.

---

### Why this plan matches your exact constraints

* Uses **only open/public datasets** (Kaggle/Grand-Challenge/SciData/official academic releases). ([Idrid][6])
* Converts segmentation → OBB systematically (not hand-wavy).
* Uses classification datasets **without pretending they have localization** (teacher distillation).
* Explicitly addresses **UWF domain shift** with real UWF corpora before supervised training. ([Nature][2])
* Uses a rotated detector stack that’s mature and supports the rotated-loss choices you need. ([GitHub][13])

If you want, paste your exact image resolution + tile size you’re currently using, and I’ll pin down concrete training hyperparameters (LRs, freeze schedule, tiling stride, augmentation set) consistent with MMRotate + ViT.

[1]: https://github.com/openmedlab/RETFound_MAE?utm_source=chatgpt.com "RETFound - A foundation model for retinal image"
[2]: https://www.nature.com/articles/s41597-024-04113-2?utm_source=chatgpt.com "Open ultrawidefield fundus image dataset with disease ..."
[3]: https://www.sciencedirect.com/science/article/pii/S2666389922001040?utm_source=chatgpt.com "DeepDRiD: Diabetic Retinopathy—Grading and Image ..."
[4]: https://csyizhou.github.io/FGADR/?utm_source=chatgpt.com "FGADR Dataset - Look Deeper into Eyes. - GitHub Pages"
[5]: https://github.com/nkicsl/DDR-dataset?utm_source=chatgpt.com "nkicsl/DDR-dataset: A General-purpose High-quality ..."
[6]: https://idrid.grand-challenge.org/?utm_source=chatgpt.com "Home - IDRiD - Grand Challenge"
[7]: https://www.adcis.net/en/third-party/e-ophtha/?utm_source=chatgpt.com "E-ophtha"
[8]: https://www.kaggle.com/datasets/nguyenhung1903/diaretdb1-v21?utm_source=chatgpt.com "DiaRetDB1 V2.1"
[9]: https://www.kaggle.com/c/diabetic-retinopathy-detection?utm_source=chatgpt.com "Diabetic Retinopathy Detection"
[10]: https://www.kaggle.com/datasets/mariaherrerot/ddrdataset?utm_source=chatgpt.com "DDR dataset"
[11]: https://academictorrents.com/details/d8653db45e7f111dc2c1b595bdac7ccf695efcfd?utm_source=chatgpt.com "APTOS 2019 diabetic retinopathy dataset"
[12]: https://odir2019.grand-challenge.org/dataset/?utm_source=chatgpt.com "Dataset-数据集 - ODIR-2019"
[13]: https://github.com/open-mmlab/mmrotate?utm_source=chatgpt.com "open-mmlab/mmrotate: OpenMMLab Rotated Object ..."
[14]: https://proceedings.mlr.press/v139/yang21l/yang21l.pdf?utm_source=chatgpt.com "Rethinking Rotated Object Detection with Gaussian ..."
[15]: https://drive.grand-challenge.org/?utm_source=chatgpt.com "DRIVE - Grand Challenge: Introduction"
[16]: https://labsites.rochester.edu/gsharma/research/computer-vision/deep-retinal-vessel-segmentation-for-ultra-widefield-fundus-photography/?utm_source=chatgpt.com "Deep Retinal Vessel Segmentation For Ultra-Widefield ..."
