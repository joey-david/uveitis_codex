#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mvcavit.data import MultiViewDataset
from mvcavit.losses import multitask_loss
from mvcavit.model import MVCAViT
from mvcavit.pso import PSO


def build_transforms(image_size, augment=False):
    ops = [transforms.Resize((image_size, image_size))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def build_loader(manifest, args, shuffle):
    dataset = MultiViewDataset(
        manifest,
        transform=build_transforms(args.image_size, augment=shuffle and args.augment),
        root=args.root,
        max_boxes=args.max_boxes,
        box_format=args.box_format,
        view_size=args.view_size,
        mirror_view=args.mirror_view,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def evaluate(model, loader, device, loss_args):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["macula"], batch["optic"])
            loss_dict = multitask_loss(outputs, batch, **loss_args)
            losses.append(loss_dict["total"].item())
    return sum(losses) / max(1, len(losses))


def parse_args():
    parser = argparse.ArgumentParser(description="Train MVCAViT on multi-view data")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--view-size", type=int, default=None)
    parser.add_argument("--max-boxes", type=int, default=10)
    parser.add_argument("--box-format", default="xyxy", choices=["xyxy", "obb"])
    parser.add_argument("--mirror-view", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--num-boxes", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--cnn-name", default="resnet18")
    parser.add_argument("--vit-name", default="vit_base_patch16_224")
    parser.add_argument("--pretrained-vit", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--obj-weight", type=float, default=0.5)
    parser.add_argument("--use-pso", action="store_true")
    parser.add_argument("--pso-iters", type=int, default=5)
    parser.add_argument("--pso-particles", type=int, default=8)
    parser.add_argument("--pso-every", type=int, default=5)
    parser.add_argument("--pso-batches", type=int, default=5)
    parser.add_argument("--pretrained", default=None, help="Path to checkpoint to load")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(args.train_manifest, args, shuffle=True)
    val_loader = build_loader(args.val_manifest, args, shuffle=False)

    model = MVCAViT(
        num_classes=args.num_classes,
        num_boxes=args.num_boxes,
        embed_dim=args.embed_dim,
        img_size=args.image_size,
        cnn_name=args.cnn_name,
        vit_name=args.vit_name,
        pretrained_vit=args.pretrained_vit,
    )
    if args.pretrained:
        state = torch.load(args.pretrained, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_args = {
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "obj_weight": args.obj_weight,
        "l2_params": list(model.parameters()),
    }

    pso = None
    if args.use_pso:
        pso = PSO(dim=2, particles=args.pso_particles)

    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["macula"], batch["optic"])
            loss_dict = multitask_loss(outputs, batch, **loss_args)
            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()
            running += loss_dict["total"].item()
        train_loss = running / max(1, len(train_loader))
        val_loss = evaluate(model, val_loader, device, loss_args)

        if pso and epoch % args.pso_every == 0:
            def fitness_fn(weights):
                model.set_fusion_weights(weights)
                batches = 0
                total = 0.0
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(batch["macula"], batch["optic"])
                        loss_dict = multitask_loss(outputs, batch, **loss_args)
                        total += loss_dict["total"].item()
                        batches += 1
                        if batches >= args.pso_batches:
                            break
                return total / max(1, batches)

            weights = pso.optimize(fitness_fn, iters=args.pso_iters)
            model.set_fusion_weights(weights)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(record)
        print(json.dumps(record))

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, output_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict()}, output_dir / "best.pt")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
