import argparse
from pathlib import Path

from utils import ensure_dir, logger

RAW_DIR = Path('datasets/raw')
DATASET_NAME = 'Uveitis-DISP'
NOTE = (
    'Place your UWF uveitis images and labels under '
    'datasets/raw/Uveitis-DISP/Images and datasets/raw/Uveitis-DISP/Labels.'
)


def download(dry_run=False):
    target_dir = RAW_DIR / DATASET_NAME
    ensure_dir(target_dir)

    if dry_run:
        logger.info('Would prepare %s -> %s', DATASET_NAME, target_dir)
        return

    logger.warning('Manual download required for %s: %s', DATASET_NAME, NOTE)
    logger.info('%s: downloaded 0.0 MB so far (manual dataset)', DATASET_NAME)


def main():
    parser = argparse.ArgumentParser(description='Prepare internal Uveitis UWF OBB dataset')
    parser.add_argument('--dry-run', action='store_true', help='Print actions without downloading')
    args = parser.parse_args()

    download(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
