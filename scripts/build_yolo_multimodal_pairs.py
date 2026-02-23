#!/usr/bin/env python3
"""Build a YOLO dataset with train-time paired modality views (Self-FI style proxy)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import yaml


def _ensure_dir(path: Path) -> None:
    """Create a directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def _symlink_rel(src: Path, dst: Path) -> None:
    """Create a relative symlink, replacing existing destination."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(rel)


def _load_ref_rgb(ref_json: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load target RGB mean/std from regular-fundus reference stats json."""
    import json

    data = json.loads(ref_json.read_text(encoding="utf-8"))
    mean_rgb = np.asarray(data["rgb_mean"], dtype=np.float32)
    std_rgb = np.asarray(data["rgb_std"], dtype=np.float32)
    return mean_rgb, std_rgb


def _modality_proxy(bgr: np.ndarray, mean_rgb: np.ndarray, std_rgb: np.ndarray, clip_limit: float) -> np.ndarray:
    """Apply a deterministic CFI-like proxy transform on non-black ROI pixels."""
    out = bgr.copy()
    roi = (out[..., 0] + out[..., 1] + out[..., 2]) > 5
    if not np.any(roi):
        return out

    # Local contrast boost in LAB while preserving color structure.
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Match ROI channel stats to regular fundus reference (BGR<->RGB conversion).
    pix = out[roi][:, ::-1].astype(np.float32)  # RGB
    src_mean = pix.mean(axis=0)
    src_std = pix.std(axis=0) + 1e-6
    pix = (pix - src_mean) / src_std * std_rgb + mean_rgb
    pix = np.clip(pix, 0, 255).astype(np.uint8)
    out2 = out.copy()
    out2[roi] = pix[:, ::-1]  # back to BGR
    return out2


def _copy_split(src: Path, out: Path, split: str, mean_rgb: np.ndarray, std_rgb: np.ndarray, clip_limit: float) -> tuple[int, int]:
    """Write one split and return (original_count, modality_count)."""
    src_img = src / "images" / split
    src_lbl = src / "labels" / split
    out_img = out / "images" / split
    out_lbl = out / "labels" / split
    _ensure_dir(out_img)
    _ensure_dir(out_lbl)

    originals = 0
    modality = 0
    for p in sorted(src_img.glob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        stem = p.stem
        lbl_src = src_lbl / f"{stem}.txt"
        if not lbl_src.exists():
            continue
        dst_img = out_img / p.name
        dst_lbl = out_lbl / lbl_src.name
        _symlink_rel(p, dst_img)
        _symlink_rel(lbl_src, dst_lbl)
        originals += 1

        if split != "train":
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        mm = _modality_proxy(img, mean_rgb=mean_rgb, std_rgb=std_rgb, clip_limit=clip_limit)
        mm_name = f"{stem}__mm{p.suffix.lower()}"
        mm_lbl = f"{stem}__mm.txt"
        cv2.imwrite(str(out_img / mm_name), mm)
        _symlink_rel(lbl_src, out_lbl / mm_lbl)
        modality += 1

    return originals, modality


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Create multimodal-pair YOLO dataset by duplicating train tiles with a CFI-like proxy view.")
    ap.add_argument("--src", type=Path, required=True, help="Source YOLO dataset root with data.yaml, images/*, labels/*.")
    ap.add_argument("--out", type=Path, required=True, help="Output YOLO dataset root.")
    ap.add_argument(
        "--ref-stats",
        type=Path,
        default=Path("preproc/ref/regular_fundus_color_stats.json"),
        help="Regular fundus color reference stats JSON.",
    )
    ap.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit for proxy modality.")
    args = ap.parse_args()

    mean_rgb, std_rgb = _load_ref_rgb(args.ref_stats)
    _ensure_dir(args.out)

    src_yaml = yaml.safe_load((args.src / "data.yaml").read_text(encoding="utf-8"))
    names = src_yaml["names"]

    counts = {}
    for split in ("train", "val", "test"):
        if not (args.src / "images" / split).exists():
            continue
        counts[split] = _copy_split(
            src=args.src,
            out=args.out,
            split=split,
            mean_rgb=mean_rgb,
            std_rgb=std_rgb,
            clip_limit=float(args.clip_limit),
        )

    data_yaml = {
        "path": args.out.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    if (args.out / "images" / "test").exists():
        data_yaml["test"] = "images/test"
    (args.out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    print({"out": args.out.as_posix(), "counts": counts})


if __name__ == "__main__":
    main()
