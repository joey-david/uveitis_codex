#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _symlink_rel(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    rel = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(rel)


def _copy_yaml(src_yaml: Path, dst_yaml: Path, out_root: Path) -> None:
    # Keep it simple: copy yaml and replace the `path:` line to point at the merged root.
    txt = _read_text(src_yaml).splitlines()
    out = []
    for line in txt:
        if line.startswith("path:"):
            out.append(f"path: {out_root.as_posix()}")
        else:
            out.append(line)
    dst_yaml.write_text("\n".join(out) + "\n", encoding="utf-8")


def _merge_split(src_roots: list[Path], split: str, out_root: Path) -> None:
    out_img = out_root / "images" / split
    out_lbl = out_root / "labels" / split
    _ensure_dir(out_img)
    _ensure_dir(out_lbl)

    for src_root in src_roots:
        img_dir = src_root / "images" / split
        lbl_dir = src_root / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for p in img_dir.iterdir():
            if p.is_file() or p.is_symlink():
                _symlink_rel(p, out_img / p.name)
        for p in lbl_dir.iterdir():
            if p.is_file() or p.is_symlink():
                _symlink_rel(p, out_lbl / p.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge multiple YOLO dataset roots into one by symlinking files.")
    ap.add_argument("--src", type=Path, nargs="+", required=True, help="Source dataset roots (each contains images/* + labels/*)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    out_root = args.out
    if out_root.exists():
        shutil.rmtree(out_root)
    _ensure_dir(out_root)

    src_roots = [p for p in args.src if p.exists()]
    if not src_roots:
        raise SystemExit("No valid --src roots found.")

    _copy_yaml(src_roots[0] / "data.yaml", out_root / "data.yaml", out_root)

    for split in args.splits:
        _merge_split(src_roots, split, out_root)

    n_train = len(list((out_root / "images/train").iterdir())) if (out_root / "images/train").exists() else 0
    n_val = len(list((out_root / "images/val").iterdir())) if (out_root / "images/val").exists() else 0
    print(f"Merged dataset written to {out_root} (train_images={n_train} val_images={n_val})")


if __name__ == "__main__":
    main()

