#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _symlink_rel(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    rel = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(rel)


def _copy_yaml(src_yaml: Path, dst_yaml: Path, out_root: Path) -> None:
    txt = src_yaml.read_text(encoding="utf-8").splitlines()
    out = []
    for line in txt:
        if line.startswith("path:"):
            out.append(f"path: {out_root.as_posix()}")
        else:
            out.append(line)
    dst_yaml.write_text("\n".join(out) + "\n", encoding="utf-8")


def _link_split(src_roots: list[Path], split: str, out_root: Path, repeats: dict[Path, int]) -> None:
    out_img = out_root / "images" / split
    out_lbl = out_root / "labels" / split
    _ensure_dir(out_img)
    _ensure_dir(out_lbl)

    for src_root in src_roots:
        rep = max(1, int(repeats.get(src_root, 1)))
        img_dir = src_root / "images" / split
        lbl_dir = src_root / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for p in img_dir.iterdir():
            if not (p.is_file() or p.is_symlink()):
                continue
            if rep == 1:
                _symlink_rel(p, out_img / p.name)
                continue
            stem = p.stem
            suf = p.suffix
            for i in range(rep):
                _symlink_rel(p, out_img / f"{stem}__r{i:02d}{suf}")

        for p in lbl_dir.iterdir():
            if not (p.is_file() or p.is_symlink()):
                continue
            if rep == 1:
                _symlink_rel(p, out_lbl / p.name)
                continue
            stem = p.stem
            suf = p.suffix
            for i in range(rep):
                _symlink_rel(p, out_lbl / f"{stem}__r{i:02d}{suf}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a YOLO dataset by mixing different train/val source roots (symlinks).")
    ap.add_argument("--train-src", type=Path, nargs="+", required=True)
    ap.add_argument("--val-src", type=Path, nargs="+", required=True)
    ap.add_argument(
        "--train-repeat",
        action="append",
        default=[],
        help="Optional repeats like 'out/yolo_obb/uwf700_tiles_main8=20' (repeatable).",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.out
    if out_root.exists():
        shutil.rmtree(out_root)
    _ensure_dir(out_root)

    train_roots = [p for p in args.train_src if p.exists()]
    val_roots = [p for p in args.val_src if p.exists()]
    if not train_roots:
        raise SystemExit("No valid --train-src roots found.")
    if not val_roots:
        raise SystemExit("No valid --val-src roots found.")

    repeats: dict[Path, int] = {}
    for s in args.train_repeat:
        if "=" not in s:
            raise SystemExit(f"Bad --train-repeat '{s}', expected PATH=N")
        p, n = s.rsplit("=", 1)
        repeats[Path(p)] = int(n)

    _copy_yaml(train_roots[0] / "data.yaml", out_root / "data.yaml", out_root)
    _link_split(train_roots, "train", out_root, repeats=repeats)
    _link_split(val_roots, "val", out_root, repeats={})

    n_train = len(list((out_root / "images/train").iterdir())) if (out_root / "images/train").exists() else 0
    n_val = len(list((out_root / "images/val").iterdir())) if (out_root / "images/val").exists() else 0
    print(f"Wrote {out_root} (train_images={n_train} val_images={n_val})")


if __name__ == "__main__":
    main()
