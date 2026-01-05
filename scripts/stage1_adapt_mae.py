#!/usr/bin/env python
import argparse
import os
import time
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from stage1.checkpoints import (
    encoder_state_from_mae,
    load_checkpoint,
    resolve_checkpoint,
    sanitize_state_dict,
    strip_prefix,
)
from stage1.data import ManifestDataset
from stage1.mae import mae_vit_large_patch16


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 RETFound MAE adaptation")
    parser.add_argument("--manifest", required=True, help="Unlabeled manifest JSONL")
    parser.add_argument(
        "--retfound",
        default="YukunZhou/RETFound_mae_natureCFP",
        help="RETFound checkpoint path or HF repo id",
    )
    parser.add_argument("--output-dir", default="runs/stage1", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    dataset = ManifestDataset(args.manifest, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    model = mae_vit_large_patch16()
    ckpt_path = resolve_checkpoint(args.retfound)
    if rank == 0:
        print(f"Using RETFound checkpoint: {ckpt_path}")
    state = sanitize_state_dict(strip_prefix(load_checkpoint(ckpt_path)))
    model.load_state_dict(state, strict=False)
    model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    if hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast = torch.amp.autocast
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = torch.cuda.amp.autocast
    for epoch in range(args.epochs):
        if distributed:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for images in loader:
            images = images.to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp):
                loss, _, _ = model(images, mask_ratio=args.mask_ratio)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        if rank == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"epoch={epoch} loss={avg_loss:.4f} time={time.time() - start:.1f}s")
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                model_state = model.module.state_dict() if distributed else model.state_dict()
                torch.save(
                    {"model": model_state, "epoch": epoch, "args": vars(args)},
                    output_dir / f"mae_adapt_epoch{epoch + 1}.pth",
                )
    if rank == 0:
        model_state = model.module.state_dict() if distributed else model.state_dict()
        torch.save(
            {"model": model_state, "epoch": args.epochs, "args": vars(args)},
            output_dir / "mae_adapt_last.pth",
        )
        encoder_state = encoder_state_from_mae(model_state)
        torch.save(
            {"model": encoder_state, "epoch": args.epochs, "args": vars(args)},
            output_dir / "encoder_adapted.pth",
        )
        print(f"Saved checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
