from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from mvcavit.data import write_jsonl, MultiViewDataset


def _make_image(path):
    img = Image.new("RGB", (64, 64), color=(123, 222, 64))
    img.save(path)


def test_manifest_and_loader(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    macula = img_dir / "sample_M.jpg"
    optic = img_dir / "sample_O.jpg"
    _make_image(macula)
    _make_image(optic)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        [
            {
                "macula_path": str(macula),
                "optic_path": str(optic),
                "label": 1,
                "boxes": [[10, 10, 20, 20]],
            }
        ],
        manifest,
    )
    ds = MultiViewDataset(manifest, transform=transforms.ToTensor(), max_boxes=5)
    loader = DataLoader(ds, batch_size=1)
    batch = next(iter(loader))
    assert isinstance(batch["macula"], torch.Tensor)
    assert batch["macula"].shape[-2:] == (64, 64)
    assert batch["boxes"].shape == (1, 5, 4)
