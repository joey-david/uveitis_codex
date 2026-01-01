# Datasets used in the RETFound → UWF uveitis OBB pipeline

## UWF-700 (Open ultrawidefield fundus image dataset with disease diagnosis & image quality assessment)
**Used for:** Stage 1 (self-supervised MAE continuation) to adapt RETFound to ultra-wide-field appearance (periphery, illumination, artifacts).  
**Why it matters:** This is the “UWF-native” anchor dataset for reducing the 45°→200° domain gap before any supervised training.  
**Get it:** Springer Nature Figshare record (article ID `26936446`). Download via Figshare API (fully scriptable).  
- Script: `python download_uwf700.py` (downloads into `datasets/raw/UWF-700/`).  
- Landing page: https://springernature.figshare.com/articles/dataset/Open_ultrawidefield_fundus_image_dataset_with_disease_diagnosis_and_clinical_image_quality_assessment/26936446  
- CLI/Python: query metadata + download each `download_url` (see Figshare API docs): https://info.figshare.com/user-guide/how-to-use-the-figshare-api/

## DeepDRiD (regular fundus + UWF; grading + quality)
**Used for:** Stage 1 (add DeepDRiD-UWF images into MAE continuation) and Stage 5 (train/fine-tune the DR classifier teacher on regular fundus, then adapt teacher on DeepDRiD-UWF).  
**Why it matters:** It provides both standard fundus and UWF distributions under one dataset, making it ideal for domain-bridging and teacher UWF-competency.  
**Get it:** GitHub repo + releases (scriptable via `git` or GitHub CLI).  
- Script: `python download_deepdrid.py` (downloads into `datasets/raw/DeepDRiD/`).  
- Repo: https://github.com/deepdrdoc/DeepDRiD  
- Releases: https://github.com/deepdrdoc/DeepDRiD/releases  
- Example (GitHub CLI): `gh release download v1.1 -R deepdrdoc/DeepDRiD`

## FGADR Seg-set (Fine-Grained Annotated Diabetic Retinopathy)
**Used for:** Stage 2 (lesion segmentation pretraining on masks), Stage 3 (convert masks→OBBs), Stage 4 (pretrain Oriented R-CNN on FGADR-derived OBBs).  
**Why it matters:** Dense lesion masks teach morphology and boundaries; converting them to rotated boxes aligns pretraining with your final OBB objective.  
**Get it:** Access-controlled but non-commercial research friendly: sign and email the form; you receive a download link (then `wget`/`curl` works).  
- Script: `python download_fgadr.py --manual` (prints access instructions; place data in `datasets/raw/FGADR/`).  
- Landing page / access instructions: https://csyizhou.github.io/FGADR/

## EyePACS / Kaggle Diabetic Retinopathy Detection (DR grading at scale)
**Used for:** Stage 5 (train the classifier teacher at scale) and Stage 6 (distill teacher probabilities into the uveitis detector during fine-tuning).  
**Why it matters:** Huge image-level supervision; distillation injects “what abnormal looks like” without needing localization labels.  
**Get it:** Kaggle competition data (scriptable via Kaggle API).  
- Script: `python download_eyepacs.py` (downloads into `datasets/raw/EyePACS/`).  
- Competition: https://www.kaggle.com/c/diabetic-retinopathy-detection  
- Example (Kaggle CLI): `kaggle competitions download -c diabetic-retinopathy-detection`

## Your Uveitis UWF OBB dataset (internal)
**Used for:** Stage 1 (as unlabeled images in MAE continuation) and Stage 6 (final Oriented R-CNN fine-tuning + evaluation).  
**Why it matters:** This is the target distribution; it calibrates symptom classes and reduces residual domain mismatch.  
**Get it:** internal / private dataset (no public link).  
- Script: `python download_uveitis_internal.py` (prints where to place data under `datasets/raw/Uveitis-DISP/`).  
