import os
import shutil
import argparse
import pandas as pd
from pathlib import Path
from utils import ensure_dir, logger

# Constants for project structure
RAW_DIR = Path('datasets/raw')
PROCESSED_DIR = Path('datasets/processed')

def standardize_uwf_700(source_dir, dest_dir):
    """Standardize UWF-700 dataset."""
    logger.info(f"Standardizing UWF-700 from {source_dir} to {dest_dir}")
    # Structure: 7 folders (Normal + 6 diseases)
    # Target: images/train (all images), labels/classification/labels.csv
    
    images_dir = dest_dir / 'images'
    ensure_dir(images_dir)
    
    records = []
    
    # Iterate through subdirectories
    for subdir in source_dir.iterdir():
        if subdir.is_dir():
            disease_label = subdir.name
            for img_file in subdir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg', '.tif']:
                    # Copy image
                    shutil.copy2(img_file, images_dir / img_file.name)
                    
                    # Record label
                    records.append({
                        'filename': img_file.name,
                        'label': disease_label,
                        'split': 'train' # UWF-700 is usually just one blob, user defines split later or here
                    })
    
    # Save labels
    if records:
        df = pd.DataFrame(records)
        labels_dir = dest_dir / 'labels' / 'classification'
        ensure_dir(labels_dir)
        df.to_csv(labels_dir / 'labels.csv', index=False)



def standardize_kaggle_dataset(source_dir, dest_dir, dataset_name):
    """Standardize Kaggle datasets (EyePACS, APTOS, etc.)."""
    logger.info(f"Standardizing {dataset_name}")
    
    # Common pattern: train_images/, test_images/, train.csv
    
    # Train
    src_train = source_dir / 'train_images' if (source_dir / 'train_images').exists() else source_dir / 'train'
    if src_train.exists():
        target_train = dest_dir / 'images' / 'train'
        ensure_dir(target_train)
        # Link or copy (copy is safer but slower, link saves space)
        # For data prep, often symbolic links are preferred if on same fs
        for img in src_train.glob('*'):
             # shutil.copy2(img, target_train / img.name) 
             # Use symlink for large datasets to save space in this example? 
             # Let's use copy for safety as requested "reproduction", but maybe symlink is better for local dev.
             # User asked for "stable ensemble of scripts for reproducible download and structuring".
             # On remote server, copy is fine.
             shutil.copy2(img, target_train / img.name)
             
    # Labels
    csv_file = source_dir / 'train.csv'
    if not csv_file.exists():
        csv_file = source_dir / 'trainLabels.csv'
        
    if csv_file.exists():
        labels_dir = dest_dir / 'labels' / 'classification'
        ensure_dir(labels_dir)
        shutil.copy2(csv_file, labels_dir / 'labels_train.csv')

def main():
    parser = argparse.ArgumentParser(description='Standardize datasets for Uveitis Codex')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process')
    args = parser.parse_args()

    ensure_dir(PROCESSED_DIR)

    # Dispatcher
    processors = {
        'UWF-700': standardize_uwf_700,
        'IDRiD': standardize_idrid,
        'EyePACS': lambda s, d: standardize_kaggle_dataset(s, d, 'EyePACS'),
        'APTOS-2019': lambda s, d: standardize_kaggle_dataset(s, d, 'APTOS-2019'),
        # Add others...
    }

    target_datasets = [args.dataset] if args.dataset else processors.keys()

    for name in target_datasets:
        if name not in processors:
            continue
            
        source_path = RAW_DIR / name
        dest_path = PROCESSED_DIR / name
        
        if source_path.exists():
            ensure_dir(dest_path)
            try:
                processors[name](source_path, dest_path)
                logger.info(f"Successfully processed {name}")
            except Exception as e:
                logger.error(f"Error processing {name}: {e}")
        else:
            logger.info(f"Source for {name} not found, skipping.")





def standardize_deepdrid(source_dir, dest_dir):
    """Standardize DeepDRiD dataset."""
    logger.info(f"Standardizing DeepDRiD from {source_dir}")
    
    # Structure: regular_fundus_images/, ultra-widefield_images/
    # We want to separate them or merge? Plan says "UWF subset" and "regular" are both used.
    # Let's create sub-datasets or just merge into one huge folder?
    # Recommendation: Keep them distinct or use a naming convention.
    # We will merge but prefix filenames to avoid collisions.
    
    subsets = {
        'regular': source_dir / 'regular_fundus_images',
        'uwf': source_dir / 'ultra-widefield_images'
    }
    
    img_dest = dest_dir / 'images' / 'train'
    ensure_dir(img_dest)
    
    for prefix, path in subsets.items():
        if not path.exists():
            continue
            
        # Recursive search for images
        for img_file in path.rglob('*'):
            if img_file.suffix.lower() in ['.jpg', '.png']:
                # Avoid thumbnails or system files
                if 'thumb' in img_file.name.lower(): continue
                
                new_name = f"{prefix}_{img_file.name}"
                shutil.copy2(img_file, img_dest / new_name)

def standardize_fgadr(source_dir, dest_dir):
    """Standardize FGADR dataset."""
    logger.info(f"Standardizing FGADR from {source_dir}")
    # Assumption: source_dir contains 'images' and 'masks' folders or similar
    
    # Copy images
    # Try looking for an 'Original_Images' or 'images' folder
    src_images = list(source_dir.rglob('*.jpg')) + list(source_dir.rglob('*.png'))
    # Filter out masks if they are mixed (often having _mask or something)
    src_images = [p for p in src_images if '_mask' not in p.name and 'Seg' not in p.parts]
    
    img_dest = dest_dir / 'images' / 'train'
    ensure_dir(img_dest)
    
    for img in src_images:
        shutil.copy2(img, img_dest / img.name)

def standardize_uveitis(source_dir, dest_dir):
    """Standardize User Uveitis Dataset."""
    logger.info(f"Standardizing Uveitis-DISP from {source_dir}")
    
    # Input: Images/Uveitis/*.jpg, Labels/Uveitis/*.txt
    # Output: images/train, labels/obb/train
    
    src_img_dir = source_dir / 'Images' / 'Uveitis'
    src_lbl_dir = source_dir / 'Labels' / 'Uveitis'
    
    if not src_img_dir.exists():
        logger.warning("Uveitis images not found.")
        return

    dst_img_dir = dest_dir / 'images' / 'train'
    dst_lbl_dir = dest_dir / 'labels' / 'obb' / 'train'
    ensure_dir(dst_img_dir)
    ensure_dir(dst_lbl_dir)
    
    # Copy images
    for img in src_img_dir.glob('*'):
        if img.suffix.lower() in ['.jpg', '.png']:
            shutil.copy2(img, dst_img_dir / img.name)
            
            # Look for corresponding label
            # Label format: Uveitis-XXX.txt for Uveitis-XXX.jpg
            lbl_name = img.stem + '.txt'
            lbl_file = src_lbl_dir / lbl_name
            
            if lbl_file.exists():
                shutil.copy2(lbl_file, dst_lbl_dir / lbl_name)
            else:
                logger.warning(f"No label found for {img.name}")

if __name__ == '__main__':
    # Update processors
    # Update processors
    processors = {
        'UWF-700': standardize_uwf_700,
        'EyePACS': lambda s, d: standardize_kaggle_dataset(s, d, 'EyePACS'),
        'DeepDRiD': standardize_deepdrid,
        'FGADR': standardize_fgadr,
        'Uveitis-DISP': standardize_uveitis,
    }
    
    # Main logic again (copied from original main to ensure it uses updated processors)
    # Since we are replacing the bottom of the file, we need to basically restart the main block
    main()

