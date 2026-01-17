import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from .boxes import clip_boxes, normalize_boxes, obb_to_aabb

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


def _resolve_path(path, root):
    if path is None:
        return None
    path = Path(path)
    if root and not path.is_absolute():
        path = Path(root) / path
    return path


def _load_image(path):
    return Image.open(path).convert("RGB")


def _boxes_from_record(record, box_format):
    boxes = record.get("boxes") or record.get("boxes_macula") or []
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    if box_format == "obb":
        boxes = [obb_to_aabb(box) for box in boxes]
    return torch.tensor(boxes, dtype=torch.float32)


def _crop_image_and_boxes(img, boxes, center, size):
    if center is None:
        return img, boxes
    w, h = img.size
    cx, cy = center
    half = size // 2
    left = int(cx - half)
    upper = int(cy - half)
    right = left + size
    lower = upper + size
    left = max(left, 0)
    upper = max(upper, 0)
    right = min(right, w)
    lower = min(lower, h)
    img = img.crop((left, upper, right, lower))
    if boxes.numel() == 0:
        return img, boxes
    boxes = boxes.clone()
    boxes[:, 0::2] -= left
    boxes[:, 1::2] -= upper
    boxes = clip_boxes(boxes, img.size)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return img, boxes[keep]


class MultiViewDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        transform=None,
        root=None,
        max_boxes=20,
        box_format="xyxy",
        view_size=None,
        mirror_view=False,
    ):
        self.records = list(read_jsonl(manifest_path))
        self.transform = transform
        self.root = Path(root) if root else None
        self.max_boxes = max_boxes
        self.box_format = box_format
        self.view_size = view_size
        self.mirror_view = mirror_view

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        macula_path = rec.get("macula_path") or rec.get("macula") or rec.get("path")
        optic_path = rec.get("optic_path") or rec.get("optic")
        if self.mirror_view or optic_path is None:
            optic_path = macula_path
        macula_path = _resolve_path(macula_path, self.root)
        optic_path = _resolve_path(optic_path, self.root)
        macula_img = _load_image(macula_path)
        optic_img = _load_image(optic_path)

        boxes = _boxes_from_record(rec, self.box_format)
        macula_center = rec.get("macula_center")
        optic_center = rec.get("optic_center")
        if self.view_size:
            macula_img, boxes = _crop_image_and_boxes(
                macula_img, boxes, macula_center, self.view_size
            )
            optic_img, _ = _crop_image_and_boxes(
                optic_img, torch.zeros((0, 4)), optic_center, self.view_size
            )

        macula_size = macula_img.size
        if self.transform:
            macula_img = self.transform(macula_img)
            optic_img = self.transform(optic_img)

        boxes = normalize_boxes(boxes, macula_size)
        padded = torch.zeros((self.max_boxes, 4), dtype=torch.float32)
        mask = torch.zeros((self.max_boxes,), dtype=torch.float32)
        if boxes.numel() > 0:
            n = min(self.max_boxes, boxes.shape[0])
            padded[:n] = boxes[:n]
            mask[:n] = 1.0

        label = int(rec.get("label", 0))
        return {
            "macula": macula_img,
            "optic": optic_img,
            "label": torch.tensor(label, dtype=torch.long),
            "boxes": padded,
            "box_mask": mask,
        }
