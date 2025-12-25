import os
import argparse
import logging
from pathlib import Path
from utils import ensure_dir, download_file, logger

# Configuration for datasets
DATASETS = {
    'UWF-700': {
        'type': 'direct',
        'url': 'https://figshare.com/ndownloader/files/49014559', # Updated valid link
        'filename': 'Dataset.zip'
    },
    'EyePACS': {
        'type': 'kaggle',
        'id': 'andreivann/eyepacs'
    },
    'DeepDRiD': {
        'type': 'direct',
        'url': 'https://github.com/deepdrdoc/DeepDRiD/archive/refs/tags/v1.1.zip',
        'filename': 'DeepDRiD.zip'
    },
    'FGADR': {
        'type': 'direct',
        'url': 'https://drive.usercontent.google.com/download?id=1auSQI3O5qd-hHB4U5gXOdDwa02Bj3yHS&export=download&authuser=0',
        'filename': 'FGADR.zip'
    }
}

RAW_DIR = Path('datasets/raw')

def download_kaggle_dataset(dataset_id, output_dir, limit_mb=None):
    """Download dataset using Kaggle API. limit_mb restricts download size."""
    try:
        import kaggle
    except OSError:
        logger.error("Kaggle API credentials not found. Please place kaggle.json in ~/.kaggle/")
        return

    logger.info(f"Downloading Kaggle dataset: {dataset_id}")
    
    if limit_mb:
        limit_bytes = int(limit_mb * 1024 * 1024)
        logger.info(f"Verification mode: Limiting download to ~{limit_mb} MB")
        
        # Initialize API
        api = kaggle.KaggleApi()
        api.authenticate()
        
        if '/' in dataset_id:
            # Dataset
            files = api.dataset_list_files(dataset_id).files
            downloaded_bytes = 0
            
            ensure_dir(output_dir)
            
            for file in files:
                if downloaded_bytes >= limit_bytes:
                    logger.info(f"Reached verification limit of {limit_mb} MB. Stopping.")
                    break
                    
                logger.info(f"Downloading {file.name} ({file.total_bytes} bytes)...")
                
                success = False
                # Attempt 1: Full path
                try:
                    api.dataset_download_file(dataset_id, file.name, path=output_dir, force=True, quiet=False)
                    success = True
                except Exception as e:
                    if "404" in str(e):
                         logger.warning(f"Failed to download {file.name} using full path (404).")
                         # Attempt 2: Basename
                         fname = os.path.basename(file.name)
                         if fname != file.name:
                             logger.info(f"Retrying with basename: {fname}")
                             try:
                                 api.dataset_download_file(dataset_id, fname, path=output_dir, force=True, quiet=False)
                                 success = True
                             except Exception as e2:
                                 logger.warning(f"Failed with basename: {e2}")
                    else:
                        logger.warning(f"Failed to download {file.name}: {e}")

                if success:
                    # Check actual downloaded file size
                    fpath = os.path.join(output_dir, file.name)
                    # Also check basename path if that's where it landed (Kaggle API quirks)
                    fpath_base = os.path.join(output_dir, os.path.basename(file.name))
                    
                    found_size = 0
                    if os.path.exists(fpath):
                        found_size = os.path.getsize(fpath)
                    elif os.path.exists(fpath + ".zip"):
                        found_size = os.path.getsize(fpath + ".zip")
                    elif os.path.exists(fpath_base):
                         found_size = os.path.getsize(fpath_base)
                    elif os.path.exists(fpath_base + ".zip"):
                         found_size = os.path.getsize(fpath_base + ".zip")
                    
                    if found_size > 0:
                        downloaded_bytes += found_size
                    else:
                        logger.warning("Download reported success but file not found on disk.")
                else:
                    logger.warning(f"Skipping {file.name} due to download errors during verification.")
                     
        else:
             # Competition - harder to list files granularly via some API versions, but let's try
             # For now fallback to full download for competitions or implement similar file listing if supported
             logger.warning("Verification partial download not fully supported for competitions in this script yet. Falling back to full download attempt which might be huge.")
             kaggle.api.competition_download_files(dataset_id, path=output_dir, quiet=False)

    else:
        if '/' in dataset_id:
            # It's a dataset (owner/dataset)
            kaggle.api.dataset_download_files(dataset_id, path=output_dir, quiet=False, unzip=True)
        else:
            # It's a competition
            kaggle.api.competition_download_files(dataset_id, path=output_dir, quiet=False)


def main():
    parser = argparse.ArgumentParser(description='Download datasets for Uveitis Codex')
    parser.add_argument('--dataset', type=str, help='Specific dataset to download')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be downloaded')
    parser.add_argument('--verify-size-mb', type=float, help='Limit download size in MB for verification')
    args = parser.parse_args()

    ensure_dir(RAW_DIR)

    target_datasets = [args.dataset] if args.dataset else DATASETS.keys()

    status_report = []

    for name in target_datasets:
        if name not in DATASETS:
            logger.warning(f"Unknown dataset: {name}")
            continue

        info = DATASETS[name]
        logger.info(f"Processing {name}...")
        
        status = "SKIPPED"
        note = ""

        if args.dry_run:
            logger.info(f"Would process {name} of type {info['type']}")
            status_report.append({'dataset': name, 'status': 'DRY_RUN', 'note': f"Type: {info['type']}"})
            continue

        target_path = RAW_DIR / name
        ensure_dir(target_path)

        if info['type'] == 'direct':
            if 'manual_instruction' in info:
                 logger.warning(f"Dataset {name} likely requires manual auth/download. Instruction: {info['manual_instruction']}")
                 status = "MANUAL_REQUIRED"
                 note = info['manual_instruction']
            else:
                 outfile = target_path / info['filename']
                 if outfile.exists():
                     logger.info(f"{name} already exists at {outfile}")
                     status = "EXISTS"
                 else:
                     try:
                        limit_bytes = int(args.verify_size_mb * 1024 * 1024) if args.verify_size_mb else None
                        download_file(info['url'], outfile, max_size_bytes=limit_bytes)
                        status = "SUCCESS"
                     except Exception as e:
                        logger.error(f"Failed to download {name}: {e}")
                        status = "FAILED"
                        note = str(e)

        elif info['type'] == 'kaggle':
            # Check for local kaggle.json
            if os.path.exists('kaggle.json'):
                os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

            # Check if likely already downloaded (folder not empty)
            if any(target_path.iterdir()):
                 logger.info(f"{name} seems to be populated.")
                 status = "EXISTS"
            else:
                try:
                    if info['id'] == 'andreivann/eyepacs' and not args.verify_size_mb:
                        logger.info("NOTE: The EyePACS dataset is approximately 34GB.")
                        logger.info("The download might appear frozen while Kaggle prepares the archive. Please be patient (can take hours).")
                    
                    download_kaggle_dataset(info['id'], str(target_path), limit_mb=args.verify_size_mb)
                    status = "SUCCESS"
                except Exception as e:
                    status = "FAILED"
                    # Capture the actual error message
                    err_msg = str(e)
                    if "403" in err_msg:
                        note = "403 Forbidden. You must accept competition rules at kaggle.com."
                        logger.error(f"Failed to download {name}: 403 Forbidden. Please accept the rules at https://www.kaggle.com/c/{info.get('id', 'dataset')}/rules")
                    else:
                        note = f"Error: {err_msg}"
                        logger.error(f"Failed to download {name}: {e}")

        elif info['type'] == 'manual':
            logger.info(f"Dataset {name} requires manual action.")
            logger.info(f"Instruction: {info['instruction']}")
            status = "MANUAL_REQUIRED"
            note = info['instruction']
            
        status_report.append({'dataset': name, 'status': status, 'note': note})

    print("\n" + "="*60)
    print("DOWNLOAD STATUS REPORT")
    print("="*60)
    print(f"{'Dataset':<15} | {'Status':<15} | {'Note'}")
    print("-" * 60)
    for item in status_report:
        # Truncate note
        print(f"{item['dataset']:<15} | {item['status']:<15} | {item['note']}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
