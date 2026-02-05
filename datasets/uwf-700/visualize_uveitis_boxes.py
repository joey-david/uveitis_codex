#!/usr/bin/env python3
"""Visualize Uveitis bounding boxes with next/prev controls."""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button
from PIL import Image


def load_label_map(csv_path: Path) -> dict:
    label_map = {}
    if not csv_path.exists():
        return label_map
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                num = int(row.get("number", "").strip())
            except ValueError:
                continue
            name = row.get("symptom", "").strip()
            if name:
                label_map[num] = name
    return label_map


def parse_label_file(label_path: Path):
    labels = []
    if not label_path.exists():
        return labels
    with label_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            try:
                coords = [float(p) for p in parts[1:]]
            except ValueError:
                continue
            if len(coords) < 4:
                continue
            if len(coords) % 2 != 0:
                continue
            points = list(zip(coords[0::2], coords[1::2]))
            if len(points) < 2:
                continue
            labels.append((class_id, points))
    return labels


def sort_key(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else path.stem


class UveitisViewer:
    def __init__(self, image_paths, labels_dir, label_map):
        self.image_paths = image_paths
        self.labels_dir = labels_dir
        self.label_map = label_map
        self.color_map = self.build_color_map()
        self.index = 0

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.18)

        ax_prev = self.fig.add_axes([0.25, 0.05, 0.2, 0.075])
        ax_next = self.fig.add_axes([0.55, 0.05, 0.2, 0.075])
        self.btn_prev = Button(ax_prev, "< Prev")
        self.btn_next = Button(ax_next, "Next >")
        self.btn_prev.on_clicked(self.prev)
        self.btn_next.on_clicked(self.next)

        self.info_text = self.fig.text(0.5, 0.01, "", ha="center")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.show_index(0)

    def build_color_map(self):
        names = sorted(set(self.label_map.values()))
        cmap = plt.get_cmap("tab20")
        color_map = {}
        for i, name in enumerate(names):
            color_map[name] = cmap(i % cmap.N)
        return color_map

    def color_for_label(self, label_name, class_id):
        color = self.color_map.get(label_name)
        if color is not None:
            return color
        cmap = plt.get_cmap("tab20")
        return cmap(class_id % cmap.N)

    def on_key(self, event):
        if event.key in ("left", "a"):
            self.prev(None)
        elif event.key in ("right", "d"):
            self.next(None)

    def prev(self, _event):
        self.show_index((self.index - 1) % len(self.image_paths))

    def next(self, _event):
        self.show_index((self.index + 1) % len(self.image_paths))

    def show_index(self, index):
        self.index = index
        image_path = self.image_paths[self.index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        self.ax.clear()
        self.ax.imshow(image)
        self.ax.set_axis_off()

        labels = parse_label_file(label_path)
        label_counts = Counter()

        for i, (class_id, points) in enumerate(labels):
            label_name = self.label_map.get(class_id, f"unknown_{class_id}")
            color = self.color_for_label(label_name, class_id)
            pixel_points = [(x * width, y * height) for x, y in points]
            poly = Polygon(
                pixel_points,
                closed=True,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            self.ax.add_patch(poly)

            label_counts[label_name] += 1
            min_x = min(x for x, _y in pixel_points)
            min_y = min(y for _x, y in pixel_points)
            self.ax.text(
                min_x,
                max(min_y - 6, 0),
                label_name,
                color="white",
                fontsize=9,
                bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"),
            )

        title = f"{image_path.name} ({self.index + 1}/{len(self.image_paths)})"
        self.ax.set_title(title)

        if label_counts:
            label_text = ", ".join(
                f"{name} x{count}" if count > 1 else name
                for name, count in label_counts.most_common()
            )
        else:
            label_text = "none"
        self.info_text.set_text(f"Labels: {label_text}")

        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Uveitis bounding boxes with next/prev navigation."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("Images/Uveitis"),
        help="Images directory (default: Images/Uveitis)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("Labels/Uveitis"),
        help="Labels directory (default: Labels/Uveitis)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Labels/Uveitis/symptom_count.csv"),
        help="CSV with label mapping (default: variables.csv)",
    )
    args = parser.parse_args()

    images_dir = args.images
    labels_dir = args.labels
    csv_path = args.csv

    image_paths = sorted(images_dir.glob("*.jpg"), key=sort_key)
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    label_map = load_label_map(csv_path)

    UveitisViewer(image_paths, labels_dir, label_map)
    plt.show()


if __name__ == "__main__":
    main()
