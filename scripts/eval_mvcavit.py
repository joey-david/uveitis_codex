#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mvcavit.boxes import box_iou
from mvcavit.data import MultiViewDataset
from mvcavit.model import MVCAViT


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MVCAViT")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--view-size", type=int, default=None)
    parser.add_argument("--max-boxes", type=int, default=10)
    parser.add_argument("--box-format", default="xyxy", choices=["xyxy", "obb"])
    parser.add_argument("--mirror-view", action="store_true")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--num-boxes", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--cnn-name", default="resnet18")
    parser.add_argument("--vit-name", default="vit_base_patch16_224")
    return parser.parse_args()


def mean_iou(pred_boxes, gt_boxes, gt_mask):
    batch = pred_boxes.shape[0]
    ious = []
    for idx in range(batch):
        pred = pred_boxes[idx]
        x1 = torch.min(pred[:, 0], pred[:, 2])
        y1 = torch.min(pred[:, 1], pred[:, 3])
        x2 = torch.max(pred[:, 0], pred[:, 2])
        y2 = torch.max(pred[:, 1], pred[:, 3])
        pred = torch.stack([x1, y1, x2, y2], dim=-1)
        valid = gt_mask[idx] > 0
        if valid.sum() == 0:
            continue
        gt = gt_boxes[idx][valid]
        iou = box_iou(pred, gt)
        best_iou = iou.max(dim=0).values
        ious.append(best_iou.mean().item())
    if not ious:
        return 0.0
    return sum(ious) / len(ious)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset = MultiViewDataset(
        args.manifest,
        transform=transform,
        root=args.root,
        max_boxes=args.max_boxes,
        box_format=args.box_format,
        view_size=args.view_size,
        mirror_view=args.mirror_view,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = MVCAViT(
        num_classes=args.num_classes,
        num_boxes=args.num_boxes,
        embed_dim=args.embed_dim,
        img_size=args.image_size,
        cnn_name=args.cnn_name,
        vit_name=args.vit_name,
        pretrained_vit=False,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    iou_scores = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["macula"], batch["optic"])
            preds = outputs["logits"].argmax(dim=1)
            correct += (preds == batch["label"]).sum().item()
            total += preds.numel()
            iou_scores.append(mean_iou(outputs["boxes"], batch["boxes"], batch["box_mask"]))

    metrics = {
        "accuracy": correct / max(1, total),
        "mean_iou": sum(iou_scores) / max(1, len(iou_scores)),
    }
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
