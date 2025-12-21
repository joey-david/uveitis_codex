# Datasets used in the uveitis UWF OBB-localization pipeline

This file lists the datasets we will use (or fine-tune on) across **pretraining → lesion/structure segmentation → DR multi-task training → final uveitis OBB localization**.

> Notes on “direct download links”
> - Some datasets are **truly direct** (Zenodo/DOI/GitHub-archive links).
> - Some are **non-commercial research only** and require a form / signed agreement; in those cases the link is the official access point.

---

## 1) Uveitis-DISP (Labeling of the Uveitis/ entries of OUWF)

- **Origin**: your project (“uveitis-disp”)
- **Purpose**: target-task fine-tuning and evaluation for **uveitis symptom localization** on UWF fundus images using **oriented bounding boxes (OBBs)**.
- **Size**: **100** labeled UWF images (your current description)
- **Direct download link**: c.f. OUWF*
- **Structure (as used by your loader)**
  - `Images/Uveitis/*.jpg` (or `.png`) — UWF images
  - `Labels/Uveitis/Uveitis-XXX.txt` — one file per image
  - Each label line: `cls x1 y1 x2 y2 x3 y3 x4 y4` (normalized coordinates, clockwise/ccw consistent per dataset convention)
  - Example:
    - `.../Labels/Uveitis/Uveitis-014.txt`
    - `2 0.367187 0.580766 0.552951 0.545502 0.583333 0.640276 0.368056 0.706397`

---

## 2) IDRiD (Indian Diabetic Retinopathy Image Dataset)

- **Origin**: IDRiD Challenge (Indian Institute of Technology / collaborators); distributed via Zenodo.
- **Purpose in our pipeline**
  - **Pixel-level lesion supervision** (HE/EX/SE/MA + OD) to build strong **lesion-aware representations** and a “lesion decoder” that can later be adapted to uveitis symptoms.
  - Optional: disease grading/localization splits for auxiliary tasks.
- **Size (commonly reported)**
  - Segmentation subset: **81 train + 81 test** (162 images)
  - Grading subset: **516** images total (train/test split provided)
- **Direct download links (Zenodo files)**
  - Segmentation: `https://zenodo.org/records/17219542/files/A.%20Segmentation.zip?download=1`
  - Disease grading: `https://zenodo.org/records/17219542/files/B.%20Disease%20Grading.zip?download=1`
  - Localization: `https://zenodo.org/records/17219542/files/C.%20Localization.zip?download=1`
  - Record page: `https://zenodo.org/records/17219542`
- **Typical structure (after unzip)**
  - `A. Segmentation/`
    - `1. Original Images/` (train/test splits)
    - `2. All Segmentation Groundtruths/` (per-lesion masks; usually one mask file per image per lesion)
  - `B. Disease Grading/` (images + CSV/XLS labels)
  - `C. Localization/` (images + lesion/structure localization labels, depending on split)

---

## 3) FGADR (Fine-Grained Annotated Diabetic Retinopathy)

- **Origin**: Inception Institute of Artificial Intelligence (IIAI) + collaborators; academic/non-commercial.
- **Purpose in our pipeline**
  - Strong **pixel-level lesion supervision** for: **hemorrhages (HE)**, **hard exudates (EX)**, **soft exudates (SE / cotton-wool)**, **MA**, **IRMA**, **NV**.
  - Also provides **image-level** labels for **laser marks** and **proliferative membrane** (useful proxies for your `laser_retinien` / `membrane_epiretinenne` symptoms).
- **Size**
  - **Seg-set: 1,842 images** (released)
  - **Grade-set: 1,000 images** (not always released; see access notes)
- **Direct download link**
  - Access is granted by the authors after you sign the agreement:
    - Research use agreement (RUA): `https://www.dropbox.com/scl/fi/0vwwgipxsv3j0hy8m60w1/IIAI_FGADR_Research_Use_Agreement.pdf?rlkey=yb46h1l4upe9a0kkov6hzotjs&dl=0`
    - Access instructions page: `https://csyizhou.github.io/FGADR/`
- **Structure**
  - The delivered archive typically contains:
    - image folder(s) (one image file per sample)
    - lesion masks (one or multiple mask files per sample; format depends on release)
    - metadata / grading labels (CSV or similar)
  - **Action item**: once you receive the package, lock the exact tree in `data_specs/fgadr_tree.txt` for reproducibility.

---

## 4) DeepDRiD (Deep Diabetic Retinopathy Image Dataset)

- **Origin**: DeepDRiD Challenge organizers; released on GitHub (CC-BY-SA-4.0).
- **Purpose in our pipeline**
  - **UWF + standard CFP** training data for DR grading + quality; crucial for **domain adaptation to UWF geometry** before uveitis fine-tuning.
  - Provides an “in-between” step: *lesion-aware* (from IDRiD/FGADR) → *DR grading on UWF* (DeepDRiD) → *uveitis OBB localization*.
- **Size**
  - Multiple splits across regular fundus + ultra-widefield; exact counts depend on the split folders shipped in the release (training/validation/evaluation).
- **Direct download links**
  - Repository: `https://github.com/deepdrdoc/DeepDRiD`
  - “Final version” tagged archive (zip): `https://github.com/deepdrdoc/DeepDRiD/archive/refs/tags/v1.1.zip`
- **Structure (from the official README)**
  - `regular_fundus_images/`
    - `regular-fundus-training/Images/`, `regular-fundus-training.csv`
    - `regular-fundus-validation/Images/`, `regular-fundus-validation.csv`
    - `Online-Challenge1&2-Evaluation/Images/` + upload/label files (varies)
  - `ultra-widefield_images/`
    - `ultra-widefield-training/Images/`, `ultra-widefield-training.csv`
    - `ultra-widefield-validation/Images/`, `ultra-widefield-validation.csv`
    - `Online-Challenge3-Evaluation/Images/` + upload/label files (varies)

---

## 5) OUWF / “Open ultra-widefield fundus image dataset with disease diagnosis & image quality”

- **Origin**: Scientific Data (2024) + Figshare deposition.
- **Purpose in our pipeline**
  - **UWF domain exposure** (image statistics, peripheral distortions, artifacts).
  - Auxiliary supervision: disease diagnosis + image quality assessment can help the encoder learn robust UWF features.
- **Size**: **700 UWF images**
- **Direct download link**
  - DOI landing page: `https://doi.org/10.6084/m9.figshare.26936446`
- **Structure (as described by the authors)**
  - **7 folders**: `Normal` + 6 disease categories (each **100 images**)
  - Labels typically provided via spreadsheet/metadata files (diagnosis + image-quality fields)

---

## 6) PRIME-FP20 (UWF vessel segmentation dataset)

- **Origin**: University of Rochester / collaborators; distributed via IEEE DataPort (DOI).
- **Purpose in our pipeline**
  - **Vessel segmentation** on true UWF FP to improve modeling of **vascular structures** (highly relevant for `vascularite`).
  - Also includes paired FA, which can be used for cross-modality registration/consistency training if desired.
- **Size**: **15** UWF fundus photography images (FP) with labeled vessel maps + masks; paired UWF FA also included.
- **Direct download link**
  - DOI landing page: `https://doi.org/10.21227/ctgj-1367`
- **Structure (expected contents)**
  - UWF FP images (one file per image)
  - Binary vessel maps (one per image)
  - Binary “valid field-of-view” masks (one per image)
  - Paired UWF FA images (one per FP image)

---

## 7) EyePACS (Kaggle “Diabetic Retinopathy Detection”)

- **Origin**: Kaggle competition / EyePACS.
- **Purpose in our pipeline**
  - Large-scale **DR grading classification** to strengthen the encoder and provide a strong “retinal disease” prior.
- **Size** (competition distribution)
  - Train: **35,126** images with labels
  - Test: ~**53k** unlabeled images (for competition inference)
- **Direct download link**
  - Kaggle dataset page: `https://www.kaggle.com/c/diabetic-retinopathy-detection/data`
- **Structure (Kaggle download)**
  - `train/` (images)
  - `test/` (images)
  - `trainLabels.csv` (image id → DR grade)
  - `sampleSubmission.csv`

---

## 8) APTOS 2019 (Kaggle “Blindness Detection”)

- **Origin**: Kaggle competition.
- **Purpose in our pipeline**
  - Additional DR grading data (different acquisition distribution than EyePACS; useful for robustness).
- **Size** (competition distribution)
  - Train: **3,662** images with labels
  - Test: **1,928** unlabeled images
- **Direct download link**
  - Kaggle competition data page: `https://www.kaggle.com/competitions/aptos2019-blindness-detection/data`
- **Structure**
  - `train_images/`
  - `test_images/`
  - `train.csv` (id_code → diagnosis)
  - `sample_submission.csv`

---

## 9) RFMiD 2.0 (Retinal Fundus Multi-Disease)

- **Origin**: RFMiD 2.0 release (Zenodo record).
- **Purpose in our pipeline**
  - Multi-disease **multi-label classification** to encourage broad retinal pathology sensitivity and reduce over-specialization to DR-only signals.
- **Size**: **860** images (train/val/test splits)
- **Direct download link**
  - Zip file (Zenodo): `https://zenodo.org/records/7505822/files/RFMiD2_0.zip?download=1`
  - Record page: `https://zenodo.org/records/7505822`
- **Structure (as per release notes)**
  - `Training/` (images + CSV labels)
  - `Validation/` (images + CSV labels)
  - `Test/` (images + CSV labels)

---

## 10) ODIR-5K (Ocular Disease Intelligent Recognition)

- **Origin**: OIA-ODIR project (often distributed via GitHub/Kaggle mirrors).
- **Purpose in our pipeline**
  - Broad ocular disease **multi-label classification** (adds additional supervision signals outside DR).
- **Size**: ~**5,000 patients** (typically paired left/right images → ~10,000 images; some releases vary)
- **Direct download links**
  - Official project/GitHub entry point: `https://github.com/nkicsl/OIA-ODIR`
  - (If you prefer Kaggle mirrors for easier scripting, use the official Kaggle dataset page corresponding to ODIR-5K.)
- **Structure (common in releases)**
  - `ODIR-5K/Images/` (left/right eye image files)
  - `full_df.csv` / `data.xlsx` (per-patient metadata + multi-label targets)

---

## 11) e-ophtha (MA / EX) — optional but useful for lesion masks

- **Origin**: ADCIS (academic, non-commercial; requires form).
- **Purpose in our pipeline**
  - Extra pixel-level supervision for **exudates** and **microaneurysms** (helps the lesion decoder generalize).
- **Size (commonly reported)**
  - e-ophtha-EX: **47** images with exudates + **35** without (82 total)
  - e-ophtha-MA: commonly reported as **~381** total (exact split varies by release notes)
- **Direct download link**
  - Official access page (download form): `https://www.adcis.net/en/third-party/e-ophtha/`
- **Structure**
  - Two sub-datasets: `e-ophtha-EX/` and `e-ophtha-MA/`
  - Images + expert annotations (often delivered as lesion mask(s) / overlays; exact format depends on the download package)

---

## Practical ingestion conventions (recommended)

To keep training scripts uniform across all sources, normalize everything to this internal layout:

- `data/<dataset_name>/images/<split>/*.jpg`
- `data/<dataset_name>/labels/`
  - segmentation: `masks/<split>/<class_name>/*.png` (binary masks) or `masks/<split>/*.tif` (multi-class)
  - classification: `labels_<split>.csv`
  - detection/obb: `labels_obb/<split>/*.txt` (YOLO-OBB style) + `classes.txt`

And keep an immutable copy of the original tree in:

- `data/<dataset_name>/ORIGINAL_TREE.txt`

