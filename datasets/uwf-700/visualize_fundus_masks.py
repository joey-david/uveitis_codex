#!/usr/bin/env python3
"""Visualize prompted SAM fundus masks with next/prev controls."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uveitis_pipeline.common import load_yaml, read_image
from uveitis_pipeline.preprocess import Sam2PromptMasker, SamPromptMasker, compute_roi_mask


def sort_key(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else path.stem


def overlay_boundary(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(mask.astype(np.uint8), 80, 160)
    out = image.copy()
    out[edges > 0] = [255, 0, 0]
    return out


class MaskViewer:
    def __init__(self, image_paths: list[Path], roi_cfg: dict):
        self.image_paths = image_paths
        self.roi_cfg = roi_cfg
        self.index = 0
        self.cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        self.sam_masker = None
        self.sam2_masker = None

        use_sam = (
            roi_cfg.get("method") == "sam_prompted"
            or roi_cfg.get("method_by_dataset", {}).get("uwf700") == "sam_prompted"
        )
        use_sam2 = (
            roi_cfg.get("method") == "sam2_prompted"
            or roi_cfg.get("method_by_dataset", {}).get("uwf700") == "sam2_prompted"
        )

        if use_sam:
            try:
                self.sam_masker = SamPromptMasker(roi_cfg.get("sam", {}))
            except Exception as e:
                print(f"SAM (v1) unavailable, using fallback: {e}")

        if use_sam2:
            try:
                self.sam2_masker = Sam2PromptMasker(roi_cfg.get("sam2", {}))
            except Exception as e:
                print(f"SAM2 unavailable, using fallback: {e}")

        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 6))
        plt.subplots_adjust(bottom=0.16)

        ax_prev = self.fig.add_axes([0.25, 0.03, 0.18, 0.08])
        ax_next = self.fig.add_axes([0.57, 0.03, 0.18, 0.08])
        self.btn_prev = Button(ax_prev, "< Prev")
        self.btn_next = Button(ax_next, "Next >")
        self.btn_prev.on_clicked(self.prev)
        self.btn_next.on_clicked(self.next)

        self.info_text = self.fig.text(0.5, 0.01, "", ha="center")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.show_index(0)

    def on_key(self, event):
        if event.key in ("left", "a"):
            self.prev(None)
        elif event.key in ("right", "d"):
            self.next(None)

    def prev(self, _event):
        self.show_index((self.index - 1) % len(self.image_paths))

    def next(self, _event):
        self.show_index((self.index + 1) % len(self.image_paths))

    def _get_render(self, image_path: Path):
        if image_path in self.cache:
            return self.cache[image_path]

        image = read_image(image_path)
        mask = compute_roi_mask(
            image,
            self.roi_cfg,
            dataset="uwf700",
            sam_masker=self.sam_masker,
            sam2_masker=self.sam2_masker,
        )
        masked = image.copy()
        masked[mask == 0] = 0
        boundary = overlay_boundary(image, mask)

        self.cache[image_path] = (boundary, masked, mask)
        return self.cache[image_path]

    def show_index(self, index: int):
        self.index = index
        image_path = self.image_paths[index]
        boundary, masked, mask = self._get_render(image_path)

        self.axes[0].clear()
        self.axes[0].imshow(boundary)
        self.axes[0].set_title("Raw + Fundus Boundary")
        self.axes[0].axis("off")

        self.axes[1].clear()
        self.axes[1].imshow(masked)
        self.axes[1].set_title("Masked Fundus Only")
        self.axes[1].axis("off")

        self.axes[2].clear()
        self.axes[2].imshow(mask, cmap="gray")
        self.axes[2].set_title("Binary Mask")
        self.axes[2].axis("off")

        area = float((mask > 0).mean())
        self.info_text.set_text(
            f"{image_path.name} ({index + 1}/{len(self.image_paths)}) | mask_area={area:.3f}"
        )
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="Visualize UWF fundus masks")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("datasets/uwf-700/Images/Uveitis"),
        help="Directory containing UWF images",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage0_preprocess.yaml"),
        help="Preprocess config with roi.sam prompts",
    )
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    image_paths = sorted(
        [p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}],
        key=sort_key,
    )
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    cfg = load_yaml(args.config)
    roi_cfg = cfg["roi"]

    MaskViewer(image_paths, roi_cfg)
    plt.show()


if __name__ == "__main__":
    main()
