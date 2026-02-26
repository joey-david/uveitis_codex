"""Datasets for native polygon labels used by the mask-first pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .common import read_jsonl


class NativeMaskDataset(Dataset):
    """Load image + multi-label masks from native label index jsonl files."""

    def __init__(
        self,
        index_jsonl: str | Path,
        num_classes: int,
        image_size: int = 0,
        keep_empty: bool = True,
    ):
        self.rows = read_jsonl(index_jsonl)
        if not keep_empty:
            self.rows = [r for r in self.rows if int(r.get("num_objects", 0)) > 0]
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.rows)

    def _load_mask(self, labels_path: str, width: int, height: int) -> np.ndarray:
        """Rasterize polygon labels to CxHxW mask tensor (uint8)."""
        rec = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        mask = np.zeros((self.num_classes, height, width), dtype=np.uint8)
        for obj in rec.get("objects", []):
            cid = int(obj["class_id"]) - 1
            if cid < 0 or cid >= self.num_classes:
                continue
            poly = np.array(list(zip(obj["polygon"][0::2], obj["polygon"][1::2])), dtype=np.float32)
            poly[:, 0] = np.clip(poly[:, 0] * width, 0, max(0, width - 1))
            poly[:, 1] = np.clip(poly[:, 1] * height, 0, max(0, height - 1))
            poly = np.round(poly).astype(np.int32)
            cv2.fillPoly(mask[cid], [poly], 1)
        return mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Return normalized image tensor, binary class masks, and metadata."""
        row = self.rows[idx]
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        mask = self._load_mask(row["labels_path"], width=int(row.get("width", w)), height=int(row.get("height", h)))

        if self.image_size > 0 and (h != self.image_size or w != self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            resized = np.zeros((self.num_classes, self.image_size, self.image_size), dtype=np.uint8)
            for c in range(self.num_classes):
                resized[c] = cv2.resize(mask[c], (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = resized
            h = w = self.image_size

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask).float()
        meta = {
            "record_id": row.get("record_id", str(idx)),
            "image_id": row.get("image_id", ""),
            "height": h,
            "width": w,
            "image_path": row["image_path"],
            "labels_path": row["labels_path"],
        }
        return image_t, mask_t, meta

    def build_balanced_sampler(
        self,
        mode: str = "mean_inv",
        power: float = 1.0,
        empty_weight: float = 0.25,
    ) -> WeightedRandomSampler:
        """Create a per-image sampler that upweights rare-class samples."""
        class_counts = np.zeros(self.num_classes, dtype=np.int64)
        image_classes: list[list[int]] = []
        for row in self.rows:
            rec = json.loads(Path(row["labels_path"]).read_text(encoding="utf-8"))
            classes = sorted({int(obj["class_id"]) - 1 for obj in rec.get("objects", []) if int(obj["class_id"]) >= 1})
            image_classes.append(classes)
            for c in classes:
                if 0 <= c < self.num_classes:
                    class_counts[c] += 1

        weights = []
        for classes in image_classes:
            if not classes:
                weights.append(float(empty_weight))
                continue
            inv = [1.0 / max(1, int(class_counts[c])) for c in classes if 0 <= c < self.num_classes]
            if not inv:
                weights.append(1.0)
                continue
            base = float(np.max(inv)) if mode == "max_inv" else float(np.mean(inv))
            weights.append(max(1e-6, base) ** float(power))

        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


class AdaptImageDataset(Dataset):
    """Load preprocessed images for contrastive RETFound adaptation."""

    def __init__(self, image_paths: list[str], image_size: int = 224):
        self.image_paths = image_paths
        self.image_size = int(image_size)

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.image_paths)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Apply lightweight color/geometry augmentation."""
        h, w = image.shape[:2]
        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
        if np.random.rand() < 0.2:
            image = np.ascontiguousarray(image[::-1, :])

        scale = np.random.uniform(0.75, 1.0)
        ch = max(16, int(h * scale))
        cw = max(16, int(w * scale))
        y0 = np.random.randint(0, max(1, h - ch + 1))
        x0 = np.random.randint(0, max(1, w - cw + 1))
        image = image[y0 : y0 + ch, x0 : x0 + cw]

        alpha = float(np.random.uniform(0.85, 1.15))
        beta = float(np.random.uniform(-10.0, 10.0))
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two augmented views from one image."""
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        v1 = self._augment(image)
        v2 = self._augment(image)
        t1 = torch.from_numpy(v1.transpose(2, 0, 1)).float() / 255.0
        t2 = torch.from_numpy(v2.transpose(2, 0, 1)).float() / 255.0
        return t1, t2


def mask_collate(batch: list[tuple[torch.Tensor, torch.Tensor, dict]]) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Collate native mask samples to batched tensors."""
    images, masks, meta = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(masks, dim=0), list(meta)
