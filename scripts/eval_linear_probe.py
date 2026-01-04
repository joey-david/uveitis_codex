#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from stage1.checkpoints import load_checkpoint, resolve_checkpoint, sanitize_state_dict, strip_prefix
from stage1.data import ManifestDataset
from stage1.mae import mae_vit_large_patch16


def parse_args():
    parser = argparse.ArgumentParser(description="Linear probe eval for RETFound MAE")
    parser.add_argument("--train", required=True, help="Train manifest JSONL")
    parser.add_argument("--val", required=True, help="Val manifest JSONL")
    parser.add_argument("--test", required=True, help="Test manifest JSONL")
    parser.add_argument(
        "--baseline",
        default="YukunZhou/RETFound_mae_natureCFP",
        help="Baseline checkpoint path or HF repo id",
    )
    parser.add_argument("--adapted", required=True, help="Adapted checkpoint path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="runs/linear_probe/results.json")
    return parser.parse_args()


def load_encoder(checkpoint_path, device):
    model = mae_vit_large_patch16()
    state = sanitize_state_dict(strip_prefix(load_checkpoint(checkpoint_path)))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_features(model, images):
    latent, _, _ = model.forward_encoder(images, mask_ratio=0.0)
    return latent[:, 0]


def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def run_probe(ckpt_path, train_loader, val_loader, test_loader, num_classes, device, epochs, lr):
    encoder = load_encoder(ckpt_path, device)
    head = torch.nn.Linear(encoder.patch_embed.proj.out_channels, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        head.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                feats = extract_features(encoder, images)
            logits = head(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val_acc = 0.0
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                feats = extract_features(encoder, images)
                val_acc += accuracy_from_logits(head(feats), labels)
            val_acc /= max(len(val_loader), 1)
        print(f"epoch={epoch} val_acc={val_acc:.4f}")
    head.eval()
    with torch.no_grad():
        test_acc = 0.0
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feats = extract_features(encoder, images)
            test_acc += accuracy_from_logits(head(feats), labels)
        test_acc /= max(len(test_loader), 1)
    return {"test_acc": test_acc}


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    train_ds = ManifestDataset(args.train, transform=train_tf, return_label=True)
    val_ds = ManifestDataset(args.val, transform=eval_tf, return_label=True)
    test_ds = ManifestDataset(args.test, transform=eval_tf, return_label=True)
    num_classes = max(int(rec["label"]) for rec in train_ds.records) + 1
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    baseline_path = resolve_checkpoint(args.baseline)
    adapted_path = resolve_checkpoint(args.adapted)
    print(f"Baseline checkpoint: {baseline_path}")
    print(f"Adapted checkpoint: {adapted_path}")
    results = {
        "baseline": run_probe(
            baseline_path, train_loader, val_loader, test_loader, num_classes, device, args.epochs, args.lr
        ),
        "adapted": run_probe(
            adapted_path, train_loader, val_loader, test_loader, num_classes, device, args.epochs, args.lr
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
