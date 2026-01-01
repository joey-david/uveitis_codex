import argparse
from pathlib import Path

from utils import ensure_dir, download_file, logger

RAW_DIR = Path('datasets/raw')
DATASET_NAME = 'UWF-700'
URL = 'https://figshare.com/ndownloader/files/49014559'
FILENAME = 'Dataset.zip'


def download(dry_run=False, force=False):
    target_dir = RAW_DIR / DATASET_NAME
    ensure_dir(target_dir)

    if dry_run:
        logger.info('Would download %s -> %s', DATASET_NAME, target_dir)
        return

    out_path = target_dir / FILENAME
    if out_path.exists() and not force:
        logger.info('%s already exists (use --force to re-download)', out_path)
        return

    if out_path.exists():
        out_path.unlink()
    download_file(URL, out_path, progress_label=DATASET_NAME)


def main():
    parser = argparse.ArgumentParser(description='Download UWF-700 dataset')
    parser.add_argument('--dry-run', action='store_true', help='Print actions without downloading')
    parser.add_argument('--force', action='store_true', help='Re-download archives even if they exist')
    args = parser.parse_args()

    download(dry_run=args.dry_run, force=args.force)


if __name__ == '__main__':
    main()
