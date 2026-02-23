#!/usr/bin/env python3
"""Create a sampled YOLO dataset split by symlinking a random subset of images/labels."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path


def _ensure_dir(path: Path) -> None:
    """Create a directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def _symlink_rel(src: Path, dst: Path) -> None:
    """Create a relative symlink, replacing existing destination."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(rel)


def _copy_yaml(src_yaml: Path, dst_yaml: Path, out_root: Path) -> None:
    """Copy YOLO data.yaml while patching path to sampled root."""
    lines = src_yaml.read_text(encoding="utf-8").splitlines()
    patched = [f"path: {out_root.as_posix()}" if ln.startswith("path:") else ln for ln in lines]
    dst_yaml.write_text("\n".join(patched) + "\n", encoding="utf-8")


def _sample_split(src: Path, out: Path, split: str, max_images: int | None, rng: random.Random) -> int:
    """Sample one split and return number of linked images."""
    src_img = src / "images" / split
    src_lbl = src / "labels" / split
    if not src_img.exists() or not src_lbl.exists():
        return 0

    out_img = out / "images" / split
    out_lbl = out / "labels" / split
    _ensure_dir(out_img)
    _ensure_dir(out_lbl)

    imgs = [p for p in sorted(src_img.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if max_images is not None and len(imgs) > max_images:
        imgs = sorted(rng.sample(imgs, max_images), key=lambda p: p.name)

    kept = 0
    for img in imgs:
        lbl = src_lbl / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        _symlink_rel(img, out_img / img.name)
        _symlink_rel(lbl, out_lbl / lbl.name)
        kept += 1
    return kept


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Sample a YOLO dataset into a new root (symlink-based).")
    ap.add_argument("--src", type=Path, required=True, help="Source YOLO root.")
    ap.add_argument("--out", type=Path, required=True, help="Output YOLO root.")
    ap.add_argument("--max-train", type=int, default=None, help="Max images to keep in train split.")
    ap.add_argument("--max-val", type=int, default=None, help="Max images to keep in val split.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    _ensure_dir(args.out)
    _copy_yaml(args.src / "data.yaml", args.out / "data.yaml", args.out)

    rng = random.Random(args.seed)
    n_train = _sample_split(args.src, args.out, "train", args.max_train, rng)
    n_val = _sample_split(args.src, args.out, "val", args.max_val, rng)
    print({"out": args.out.as_posix(), "train_images": n_train, "val_images": n_val})


if __name__ == "__main__":
    main()
