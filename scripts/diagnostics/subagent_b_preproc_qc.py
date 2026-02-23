#!/usr/bin/env python3
"""Preprocessing integrity checks for ROI masking and normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _stats(values: list[float]) -> dict:
    """Return compact summary stats for a numeric list."""
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _iter_pairs(roi_dir: Path, norm_dir: Path, limit: int) -> list[tuple[Path, Path]]:
    """Return aligned ROI/NORM file pairs by stem."""
    roi_files = sorted(roi_dir.glob("*.png"))
    out = []
    for roi_path in roi_files:
        norm_path = norm_dir / roi_path.name
        if norm_path.exists():
            out.append((roi_path, norm_path))
    if limit > 0:
        return out[:limit]
    return out


def _load_roi_aligned(roi_path: Path, norm_shape: tuple[int, int], crop_meta_dir: Path | None) -> tuple[np.ndarray, bool]:
    """Load ROI mask and align it to normalized image geometry."""
    roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise RuntimeError(f"Could not read ROI mask: {roi_path}")
    mismatched = False
    if crop_meta_dir is not None:
        meta_path = crop_meta_dir / f"{roi_path.stem}.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            x0, y0, x1, y1 = [int(v) for v in meta.get("bbox_xyxy", [0, 0, roi.shape[1], roi.shape[0]])]
            x0 = max(0, min(x0, roi.shape[1]))
            x1 = max(0, min(x1, roi.shape[1]))
            y0 = max(0, min(y0, roi.shape[0]))
            y1 = max(0, min(y1, roi.shape[0]))
            if x1 > x0 and y1 > y0:
                roi = roi[y0:y1, x0:x1]
    if roi.shape[:2] != norm_shape:
        mismatched = True
        roi = cv2.resize(roi, (norm_shape[1], norm_shape[0]), interpolation=cv2.INTER_NEAREST)
    return roi, mismatched


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Audit ROI mask + normalized image consistency.")
    ap.add_argument("--roi-dir", type=Path, required=True)
    ap.add_argument("--norm-dir", type=Path, required=True)
    ap.add_argument("--crop-meta-dir", type=Path, default=Path("preproc/crop_meta"))
    ap.add_argument("--sample-limit", type=int, default=0)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    pairs = _iter_pairs(args.roi_dir, args.norm_dir, args.sample_limit)
    outside_nonzero = []
    inside_black = []
    roi_coverages = []
    by_dataset: dict[str, dict[str, list[float]]] = {}
    offenders = []
    shape_mismatch = 0

    for roi_path, norm_path in pairs:
        img = cv2.imread(str(norm_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        roi, mismatched = _load_roi_aligned(roi_path, img.shape[:2], args.crop_meta_dir)
        if mismatched:
            shape_mismatch += 1

        mask = roi > 0
        outside = ~mask
        img_nonzero = np.any(img > 0, axis=2)
        outside_ratio = float(np.mean(img_nonzero[outside])) if np.any(outside) else 0.0
        inside_black_ratio = float(np.mean(~img_nonzero[mask])) if np.any(mask) else 1.0
        coverage = float(np.mean(mask))

        outside_nonzero.append(outside_ratio)
        inside_black.append(inside_black_ratio)
        roi_coverages.append(coverage)
        ds = roi_path.stem.split("__", 1)[0]
        if ds not in by_dataset:
            by_dataset[ds] = {"outside_nonzero_ratio": [], "inside_black_ratio": [], "roi_coverage": []}
        by_dataset[ds]["outside_nonzero_ratio"].append(outside_ratio)
        by_dataset[ds]["inside_black_ratio"].append(inside_black_ratio)
        by_dataset[ds]["roi_coverage"].append(coverage)

        if outside_ratio > 0.005 or inside_black_ratio > 0.2:
            offenders.append(
                {
                    "image": roi_path.name,
                    "outside_nonzero_ratio": outside_ratio,
                    "inside_black_ratio": inside_black_ratio,
                    "roi_coverage": coverage,
                }
            )

    report = {
        "num_pairs": len(pairs),
        "shape_mismatch_count": shape_mismatch,
        "outside_nonzero_ratio": _stats(outside_nonzero),
        "inside_black_ratio": _stats(inside_black),
        "roi_coverage": _stats(roi_coverages),
        "num_offenders": len(offenders),
        "top_offenders": offenders[:50],
        "per_dataset": {
            ds: {
                "outside_nonzero_ratio": _stats(vals["outside_nonzero_ratio"]),
                "inside_black_ratio": _stats(vals["inside_black_ratio"]),
                "roi_coverage": _stats(vals["roi_coverage"]),
            }
            for ds, vals in sorted(by_dataset.items())
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
