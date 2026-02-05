from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF


class CocoDetectionDataset(Dataset):
    def __init__(self, coco_json: str | Path, resize: int | None = None):
        data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
        self.images = data["images"]
        self.categories = {c["id"]: c["name"] for c in data["categories"]}
        self.resize = resize

        by_image = defaultdict(list)
        for ann in data["annotations"]:
            by_image[ann["image_id"]].append(ann)
        self.by_image = by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        record = self.images[idx]
        image = Image.open(record["file_name"]).convert("RGB")
        w0, h0 = image.size

        anns = self.by_image.get(record["id"], [])
        boxes = []
        labels = []
        areas = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]))
            areas.append(float(ann["area"]))

        if self.resize and (w0 != self.resize or h0 != self.resize):
            image = TF.resize(image, [self.resize, self.resize])
            sx = self.resize / w0
            sy = self.resize / h0
            boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]
            areas = [a * sx * sy for a in areas]

        image = TF.to_tensor(image)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            areas_t = torch.tensor(areas, dtype=torch.float32)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([int(record["id"])]),
            "area": areas_t,
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }
        return image, target

    def build_class_balanced_sampler(self) -> WeightedRandomSampler:
        class_freq = Counter()
        img_classes = []
        for record in self.images:
            anns = self.by_image.get(record["id"], [])
            classes = sorted({int(a["category_id"]) for a in anns})
            img_classes.append(classes)
            class_freq.update(classes)

        weights = []
        for classes in img_classes:
            if not classes:
                weights.append(1.0)
                continue
            w = sum(1.0 / max(class_freq[c], 1) for c in classes) / len(classes)
            weights.append(w)
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
