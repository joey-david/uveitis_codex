# Datasets used in the MVCAViT pipeline

## UWF-700 (Open ultrawidefield fundus image dataset)
**Used for:** Optional DR pretraining and domain exposure to UWF appearance.
**Get it:** Springer Nature Figshare record (article ID `26936446`).
- Script: `python datasets/individual_dataset_downloaders/download_uwf700.py`

## DeepDRiD (regular fundus + UWF; grading + quality)
**Used for:** Optional DR pretraining and validation.
**Get it:** GitHub repo + releases.
- Script: `python datasets/individual_dataset_downloaders/download_deepdrid.py`

## FGADR (lesion segmentation masks)
**Used for:** Optional lesion morphology pretraining if you want to bootstrap box prediction.
**Get it:** Access-controlled (request link).
- Script: `python datasets/individual_dataset_downloaders/download_fgadr.py --manual`

## EyePACS (Kaggle DR grading)
**Used for:** Optional large-scale DR classification pretraining.
**Get it:** Kaggle competition data.
- Script: `python datasets/individual_dataset_downloaders/download_eyepacs.py`

## Your Uveitis UWF dataset (internal)
**Used for:** Final fine-tuning and evaluation for uveitis bounding boxes.
**Get it:** internal / private dataset.
- Script: `python datasets/individual_dataset_downloaders/download_uveitis_internal.py`

## Two-field DR datasets (macula + optic disc)
If you have DRTiD or another paired-view dataset, use it directly with
`scripts/build_multiview_manifest.py` to match the MVCAViT multi-view input format.
