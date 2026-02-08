from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2

from .common import read_image, read_jsonl, save_json


def report_dataset(manifest_paths: list[str], coco_paths: list[str], out_json: str) -> dict:
    counts = {"images": 0, "datasets": Counter(), "label_formats": Counter(), "splits": Counter()}
    for path in manifest_paths:
        for row in read_jsonl(path):
            counts["images"] += 1
            counts["datasets"][row["dataset"]] += 1
            counts["label_formats"][row["label_format"]] += 1
            counts["splits"][row["split"]] += 1

    coco_stats = {}
    for path in coco_paths:
        if not Path(path).exists():
            continue
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        class_counts = Counter(a["category_id"] for a in data["annotations"])
        areas = [a["area"] for a in data["annotations"]]
        coco_stats[Path(path).name] = {
            "images": len(data["images"]),
            "annotations": len(data["annotations"]),
            "class_counts": dict(class_counts),
            "avg_lesion_area": float(np.mean(areas)) if areas else 0.0,
        }

    out = {
        "images": counts["images"],
        "datasets": dict(counts["datasets"]),
        "label_formats": dict(counts["label_formats"]),
        "splits": dict(counts["splits"]),
        "coco": coco_stats,
    }
    save_json(out_json, out)
    return out


def report_preproc(
    manifest_path: str,
    preproc_root: str,
    out_dir: str,
    sample_n: int = 24,
) -> dict:
    preproc = Path(preproc_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(manifest_path)
    random.seed(42)
    sample = random.sample(rows, k=min(sample_n, len(rows)))

    before = []
    after = []
    roi_areas = []

    fig, axes = plt.subplots(4, 6, figsize=(16, 10))
    axes = axes.flatten()

    for i, row in enumerate(sample):
        key = row["image_id"].replace("::", "__")
        raw = read_image(row["filepath"])
        norm_path = preproc / "norm" / f"{key}.png"
        if not norm_path.exists():
            continue
        norm = read_image(norm_path)
        if norm.shape[:2] != raw.shape[:2]:
            norm = cv2.resize(norm, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)

        before.append(raw.reshape(-1, 3))
        after.append(norm.reshape(-1, 3))

        mask_path = preproc / "roi_masks" / f"{key}.png"
        if mask_path.exists():
            mask = read_image(mask_path)[:, :, 0] > 0
            roi_areas.append(float(mask.mean()))

        if i < len(axes):
            axes[i].imshow(np.hstack([raw, norm]))
            axes[i].set_title(key, fontsize=8)
            axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(out / "raw_vs_norm_grid.png", dpi=180)
    plt.close(fig)

    if before and after:
        before_arr = np.concatenate(before, axis=0)
        after_arr = np.concatenate(after, axis=0)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        channels = ["R", "G", "B"]
        for c in range(3):
            axes[c].hist(before_arr[:, c], bins=64, alpha=0.5, label="before")
            axes[c].hist(after_arr[:, c], bins=64, alpha=0.5, label="after")
            axes[c].set_title(channels[c])
        axes[0].legend()
        plt.tight_layout()
        plt.savefig(out / "roi_hist_before_after.png", dpi=180)
        plt.close(fig)

    report = {
        "num_samples": len(sample),
        "avg_roi_area_ratio": float(np.mean(roi_areas)) if roi_areas else 0.0,
        "grid": (out / "raw_vs_norm_grid.png").as_posix(),
        "hist": (out / "roi_hist_before_after.png").as_posix(),
    }
    save_json(out / "preproc_report.json", report)
    return report


def report_training(run_dir: str, out_json: str, out_png: str) -> dict:
    run = Path(run_dir)
    metrics_path = run / "metrics.jsonl"
    rows = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(line))

    if rows:
        epochs = [r["epoch"] for r in rows]
        train_loss = [r["train_loss"] for r in rows]
        val_map = [r["val_mAP_proxy"] for r in rows]

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(epochs, train_loss, label="train_loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_map, color="orange", label="val_mAP_proxy")
        ax2.set_ylabel("mAP_proxy")
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close(fig)

    best_report_path = run / "val_report.json"
    best = json.loads(best_report_path.read_text(encoding="utf-8")) if best_report_path.exists() else {}

    out = {
        "n_epochs": len(rows),
        "best": best,
        "curves_png": out_png,
    }
    save_json(out_json, out)
    return out


def ablate_preproc(pred_dir_a: str, pred_dir_b: str, out_json: str) -> dict:
    pa = Path(pred_dir_a)
    pb = Path(pred_dir_b)
    shared = sorted({p.name for p in pa.glob("*.json")} & {p.name for p in pb.glob("*.json")})

    deltas = []
    for name in shared:
        a = json.loads((pa / name).read_text(encoding="utf-8"))
        b = json.loads((pb / name).read_text(encoding="utf-8"))
        na = len(a.get("predictions", []))
        nb = len(b.get("predictions", []))
        deltas.append(nb - na)

    out = {
        "num_images": len(shared),
        "avg_prediction_count_delta": float(np.mean(deltas)) if deltas else 0.0,
    }
    save_json(out_json, out)
    return out
