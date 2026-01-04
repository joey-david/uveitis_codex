from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from stage1.data import find_images, write_jsonl, ManifestDataset


def _make_image(path):
    img = Image.new("RGB", (64, 64), color=(123, 222, 64))
    img.save(path)


def test_manifest_and_loader(tmp_path):
    img_dir = tmp_path / "images" / "classA"
    img_dir.mkdir(parents=True)
    for idx in range(4):
        _make_image(img_dir / f"img_{idx}.jpg")
    paths = find_images([img_dir])
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl([{"path": str(p)} for p in paths], manifest)
    ds = ManifestDataset(manifest, transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == 2
