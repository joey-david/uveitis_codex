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


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter a YOLO dataset split by whether it has labels (symlinks).")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--mode", choices=["pos", "bg"], required=True, help="pos=keep labeled images, bg=keep empty-label images")
    args = ap.parse_args()

    src = args.src
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    _ensure_dir(out)
    _copy_yaml(src / "data.yaml", out / "data.yaml", out)

    src_img = src / "images" / args.split
    src_lbl = src / "labels" / args.split
    out_img = out / "images" / args.split
    out_lbl = out / "labels" / args.split
    _ensure_dir(out_img)
    _ensure_dir(out_lbl)

    kept = 0
    for lbl in sorted(src_lbl.glob("*.txt")):
        has = bool(lbl.read_text(encoding="utf-8").strip())
        if (args.mode == "pos" and not has) or (args.mode == "bg" and has):
            continue
        img = src_img / (lbl.stem + ".png")
        if not img.exists():
            # allow jpg too
            cands = list(src_img.glob(lbl.stem + ".*"))
            img = cands[0] if cands else None
        if not img or not Path(img).exists():
            continue
        _symlink_rel(Path(img), out_img / Path(img).name)
        _symlink_rel(lbl, out_lbl / lbl.name)
        kept += 1

    print(f"Wrote {out} split={args.split} mode={args.mode} kept={kept}")


if __name__ == "__main__":
    main()

