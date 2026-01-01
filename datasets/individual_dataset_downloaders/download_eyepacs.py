import argparse
import os
import threading
from pathlib import Path

from utils import ensure_dir, logger

RAW_DIR = Path('datasets/raw')
DATASET_NAME = 'EyePACS'
COMPETITION = 'diabetic-retinopathy-detection'
_MB = 1024 * 1024


def _format_bytes(num_bytes):
    return f'{num_bytes / _MB:.1f} MB'


def _dir_size(path):
    total = 0
    for entry in path.iterdir():
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _start_progress_watcher(target_dir, label, baseline_bytes):
    stop_event = threading.Event()

    def _watch():
        while not stop_event.is_set():
            current = _dir_size(target_dir)
            downloaded = max(0, current - baseline_bytes)
            logger.info('%s: downloaded %s so far', label, _format_bytes(downloaded))
            stop_event.wait(2.0)

    thread = threading.Thread(target=_watch, daemon=True)
    thread.start()
    return stop_event, thread


def download(dry_run=False, force=False):
    target_dir = RAW_DIR / DATASET_NAME
    ensure_dir(target_dir)

    if dry_run:
        logger.info('Would download %s -> %s', DATASET_NAME, target_dir)
        return

    if Path('kaggle.json').exists():
        os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

    baseline_bytes = _dir_size(target_dir)
    stop_event, thread = _start_progress_watcher(target_dir, DATASET_NAME, baseline_bytes)
    try:
        import kaggle

        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            COMPETITION,
            path=str(target_dir),
            quiet=False,
            force=force,
        )
        stop_event.set()
        thread.join()
        downloaded = max(0, _dir_size(target_dir) - baseline_bytes)
        logger.info('%s: downloaded %s total', DATASET_NAME, _format_bytes(downloaded))
    except Exception as exc:
        stop_event.set()
        thread.join()
        if '403' in str(exc):
            logger.error(
                'Kaggle 403 for %s: accept the competition rules/license on Kaggle, then retry.',
                COMPETITION,
            )
        else:
            logger.error('Kaggle download failed for %s: %s', COMPETITION, exc)


def main():
    parser = argparse.ArgumentParser(description='Download EyePACS Kaggle dataset')
    parser.add_argument('--dry-run', action='store_true', help='Print actions without downloading')
    parser.add_argument('--force', action='store_true', help='Re-download archives even if they exist')
    args = parser.parse_args()

    download(dry_run=args.dry_run, force=args.force)


if __name__ == '__main__':
    main()
