import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(roots):
    paths = []
    for root in roots:
        root_path = Path(root)
        for path in root_path.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTS:
                paths.append(path)
    return paths


def write_jsonl(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def make_relative(path, rel_to):
    path = Path(path).resolve()
    rel_to = Path(rel_to).resolve()
    if rel_to in path.parents:
        return str(path.relative_to(rel_to))
    return str(path)


class ManifestDataset(Dataset):
    def __init__(self, manifest_path, transform=None, root=None, return_label=False):
        self.records = list(read_jsonl(manifest_path))
        self.transform = transform
        self.root = Path(root) if root else None
        self.return_label = return_label

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        path = Path(rec["path"])
        if self.root and not path.is_absolute():
            path = self.root / path
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.return_label:
            return img, int(rec["label"])
        return img
