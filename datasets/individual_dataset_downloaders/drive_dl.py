# pip install -U gdown
import gdown
from pathlib import Path
import sys
import threading
import time

FILE_ID = "1lDhQSTA8ffGJSIhlhrk1OJZEnwtk3gNX"
OUT = Path("datasets_bundle.zip")


def _download():
    gdown.download(id=FILE_ID, output=str(OUT), quiet=True)


download_thread = threading.Thread(target=_download, daemon=True)
download_thread.start()

last_size = -1
while download_thread.is_alive():
    if OUT.exists():
        size = OUT.stat().st_size
        if size != last_size:
            sys.stdout.write(f"\rDownloaded {size / (1024 * 1024):.1f} MB")
            sys.stdout.flush()
            last_size = size
    time.sleep(0.2)

download_thread.join()
if OUT.exists():
    size = OUT.stat().st_size
    sys.stdout.write(f"\rDownloaded {size / (1024 * 1024):.1f} MB\n")
