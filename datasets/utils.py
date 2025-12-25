import os
import requests
import hashlib
import logging
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('uveitis_dataset_prep')

def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def verify_checksum(file_path, expected_md5):
    """Verify MD5 checksum of a file."""
    if not os.path.exists(file_path):
        return False
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest() == expected_md5

def download_file(url, target_path, max_size_bytes=None):
    """Download a file with progress bar. Option to stop after max_size_bytes."""
    ensure_dir(os.path.dirname(target_path))
    
    logger.info(f"Downloading {url} to {target_path}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, stream=True, headers=headers)
    
    # 202 Accepted is common for some repositories (processing/queued), 
    # but often sends data immediately in body or requires no action if just a direct link.
    if response.status_code not in [200, 202]:
        raise Exception(f"Failed to download: Status Code {response.status_code}")
        
    total_size = int(response.headers.get('content-length', 0))
    
    block_size = 1024 # 1 Kibibyte
    
    downloaded = 0
    with open(target_path, 'wb') as file, tqdm(
        desc=os.path.basename(target_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            downloaded += size
            bar.update(size)
            
            if max_size_bytes and downloaded >= max_size_bytes:
                logger.info(f"Reached verification limit of {max_size_bytes} bytes. Stopping download.")
                break

def check_file_exists(path):
    return os.path.exists(path)
