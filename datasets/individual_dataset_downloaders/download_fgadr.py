import argparse
from pathlib import Path

from utils import ensure_dir, download_gdrive, logger

RAW_DIR = Path('datasets/raw')
DATASET_NAME = 'FGADR'
DEFAULT_GDRIVE_ID = '1auSQI3O5qd-hHB4U5gXOdDwa02Bj3yHS'
FILENAME = 'FGADR.zip'
MANUAL_NOTE = (
    'Access-controlled dataset: follow https://csyizhou.github.io/FGADR/ '
    'to request access, then download into datasets/raw/FGADR.'
)


def download(dry_run=False, force=False, gdrive_id=DEFAULT_GDRIVE_ID, manual=False):
    target_dir = RAW_DIR / DATASET_NAME
    ensure_dir(target_dir)

    if manual:
        logger.warning('Manual download required for %s: %s', DATASET_NAME, MANUAL_NOTE)
        return

    if dry_run:
        logger.info('Would download %s -> %s', DATASET_NAME, target_dir)
        return

    out_path = target_dir / FILENAME
    if out_path.exists() and not force:
        logger.info('%s already exists (use --force to re-download)', out_path)
        return

    if out_path.exists():
        out_path.unlink()
    download_gdrive(gdrive_id, out_path, progress_label=DATASET_NAME)


def main():
    parser = argparse.ArgumentParser(description='Download FGADR Seg-set dataset')
    parser.add_argument('--dry-run', action='store_true', help='Print actions without downloading')
    parser.add_argument('--force', action='store_true', help='Re-download archives even if they exist')
    parser.add_argument('--manual', action='store_true', help='Skip download and print manual access instructions')
    parser.add_argument(
        '--gdrive-id',
        default=DEFAULT_GDRIVE_ID,
        help='Google Drive file id (override if you received a different link)',
    )
    args = parser.parse_args()

    download(
        dry_run=args.dry_run,
        force=args.force,
        gdrive_id=args.gdrive_id,
        manual=args.manual,
    )


if __name__ == '__main__':
    main()
