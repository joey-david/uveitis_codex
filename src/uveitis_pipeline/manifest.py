from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd

from .common import parse_eye_token, save_json, save_jsonl


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _image_hw(path: Path, read_size: bool = True) -> tuple[int, int]:
    if not read_size:
        return 0, 0
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return w, h


def _safe_rel(path: Path) -> str:
    return path.as_posix()


def _scan_uwf700(root: Path, read_size: bool) -> list[dict]:
    rows: list[dict] = []
    labels_root = root / "Labels" / "Uveitis"
    for disease_dir in sorted((root / "Images").glob("*")):
        if not disease_dir.is_dir():
            continue
        disease = disease_dir.name
        for img_path in sorted(disease_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            width, height = _image_hw(img_path, read_size=read_size)
            label_path = labels_root / f"{img_path.stem}.txt"
            rows.append(
                {
                    "image_id": f"uwf700::{img_path.stem}",
                    "filepath": _safe_rel(img_path),
                    "dataset": "uwf700",
                    "split": "",
                    "eye": parse_eye_token(img_path.stem),
                    "labels_path": _safe_rel(label_path) if label_path.exists() else "",
                    "label_format": "obb" if label_path.exists() else "none",
                    "width": width,
                    "height": height,
                    "patient_id": img_path.stem,
                    "notes": f"disease={disease}",
                }
            )
    return rows


def _scan_fgadr(root: Path, read_size: bool) -> list[dict]:
    rows: list[dict] = []
    mask_dirs = sorted(root.glob("*_Masks"))
    for img_path in sorted((root / "Original_Images").glob("*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        width, height = _image_hw(img_path, read_size=read_size)
        mask_hits = [d / img_path.name for d in mask_dirs if (d / img_path.name).exists()]
        rows.append(
            {
                "image_id": f"fgadr::{img_path.stem}",
                "filepath": _safe_rel(img_path),
                "dataset": "fgadr",
                "split": "",
                "eye": parse_eye_token(img_path.stem),
                "labels_path": _safe_rel(root),
                "label_format": "mask",
                "width": width,
                "height": height,
                "patient_id": img_path.stem,
                "notes": f"mask_count={len(mask_hits)}",
            }
        )
    return rows


def _parse_deepdrid_csv(csv_path: Path, dataset_tag: str, split: str, read_size: bool) -> list[dict]:
    rows: list[dict] = []
    if not csv_path.exists():
        return rows
    base = csv_path.parent
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for item in reader:
            image_path = item.get("image_path", "").replace("\\", "/").strip("/")
            if image_path:
                resolved = base / image_path
            else:
                image_id = item.get("image_id", "").strip()
                resolved = base / "Images" / image_id.split("_")[0] / f"{image_id}.jpg"
            if not resolved.exists():
                continue
            width, height = _image_hw(resolved, read_size=read_size)
            dr_level = item.get("DR_level") or item.get("DR_Level") or item.get("patient_DR_Level")
            eye = item.get("position", "")
            eye = "L" if "left" in eye.lower() else "R" if "right" in eye.lower() else parse_eye_token(resolved.stem)
            rows.append(
                {
                    "image_id": f"{dataset_tag}::{resolved.stem}",
                    "filepath": _safe_rel(resolved),
                    "dataset": dataset_tag,
                    "split": split,
                    "eye": eye,
                    "labels_path": _safe_rel(csv_path),
                    "label_format": "grade",
                    "width": width,
                    "height": height,
                    "patient_id": str(item.get("patient_id") or resolved.stem.split("_")[0]),
                    "notes": f"grade={dr_level}" if dr_level not in {None, ''} else "",
                }
            )
    return rows


def _scan_deepdrid(root: Path, read_size: bool) -> list[dict]:
    rows: list[dict] = []
    rows += _parse_deepdrid_csv(
        root / "ultra-widefield_images" / "ultra-widefield-training" / "ultra-widefield-training.csv",
        "deepdrid_uwf",
        "train",
        read_size,
    )
    rows += _parse_deepdrid_csv(
        root / "ultra-widefield_images" / "ultra-widefield-validation" / "ultra-widefield-validation.csv",
        "deepdrid_uwf",
        "val",
        read_size,
    )
    rows += _parse_deepdrid_csv(
        root / "ultra-widefield_images" / "Online-Challenge3-Evaluation" / "Challenge3_upload.csv",
        "deepdrid_uwf",
        "test",
        read_size,
    )
    rows += _parse_deepdrid_csv(
        root / "regular_fundus_images" / "regular-fundus-training" / "regular-fundus-training.csv",
        "deepdrid_regular",
        "train",
        read_size,
    )
    rows += _parse_deepdrid_csv(
        root / "regular_fundus_images" / "regular-fundus-validation" / "regular-fundus-validation.csv",
        "deepdrid_regular",
        "val",
        read_size,
    )
    rows += _parse_deepdrid_csv(
        root / "regular_fundus_images" / "Online-Challenge1&2-Evaluation" / "Challenge1_upload.csv",
        "deepdrid_regular",
        "test",
        read_size,
    )
    return rows


def _scan_eyepacs(root: Path, read_size: bool) -> list[dict]:
    rows: list[dict] = []
    label_csv = root / "trainLabels.csv"
    labels = {}
    if label_csv.exists():
        df = pd.read_csv(label_csv)
        for _, r in df.iterrows():
            labels[str(r["image"])] = int(r["level"])
    for img_path in sorted((root / "train").glob("*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        width, height = _image_hw(img_path, read_size=read_size)
        grade = labels.get(img_path.stem)
        rows.append(
            {
                "image_id": f"eyepacs::{img_path.stem}",
                "filepath": _safe_rel(img_path),
                "dataset": "eyepacs",
                "split": "",
                "eye": parse_eye_token(img_path.stem),
                "labels_path": _safe_rel(label_csv) if label_csv.exists() else "",
                "label_format": "grade" if grade is not None else "none",
                "width": width,
                "height": height,
                "patient_id": img_path.stem.split("_")[0],
                "notes": f"grade={grade}" if grade is not None else "",
            }
        )
    return rows


def _assign_random_splits(rows: list[dict], ratios: dict[str, float], seed: int) -> None:
    import random

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if row["split"]:
            continue
        grouped[(row["dataset"], str(row["patient_id"]))].append(idx)

    groups_by_dataset: dict[str, list[list[int]]] = defaultdict(list)
    for (dataset, _), idxs in grouped.items():
        groups_by_dataset[dataset].append(idxs)

    for dataset, groups in groups_by_dataset.items():
        rnd = random.Random(seed + abs(hash(dataset)) % 100_000)
        rnd.shuffle(groups)
        n = len(groups)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        train_groups = groups[:n_train]
        val_groups = groups[n_train : n_train + n_val]
        test_groups = groups[n_train + n_val :]

        for g in train_groups:
            for idx in g:
                rows[idx]["split"] = "train"
        for g in val_groups:
            for idx in g:
                rows[idx]["split"] = "val"
        for g in test_groups:
            for idx in g:
                rows[idx]["split"] = "test"


def build_manifests(cfg: dict) -> tuple[dict[str, list[dict]], dict]:
    datasets_cfg = cfg["datasets"]
    read_size = bool(cfg.get("read_image_size", True))
    all_rows: dict[str, list[dict]] = {}

    if datasets_cfg.get("uwf700", {}).get("enabled", True):
        all_rows["uwf700"] = _scan_uwf700(Path(datasets_cfg["uwf700"]["root"]), read_size)
    if datasets_cfg.get("fgadr", {}).get("enabled", True):
        all_rows["fgadr"] = _scan_fgadr(Path(datasets_cfg["fgadr"]["root"]), read_size)
    if datasets_cfg.get("deepdrid", {}).get("enabled", True):
        all_rows["deepdrid"] = _scan_deepdrid(Path(datasets_cfg["deepdrid"]["root"]), read_size)
    if datasets_cfg.get("eyepacs", {}).get("enabled", True):
        all_rows["eyepacs"] = _scan_eyepacs(Path(datasets_cfg["eyepacs"]["root"]), read_size)

    merged_rows = [row for rows in all_rows.values() for row in rows]
    _assign_random_splits(
        merged_rows,
        cfg["split_ratios"],
        int(cfg.get("seed", 42)) + int(cfg.get("fold", 0)),
    )

    by_image_id = {row["image_id"]: row for row in merged_rows}
    for name, rows in all_rows.items():
        all_rows[name] = [by_image_id[row["image_id"]] for row in rows]

    split_dict = {
        "train": [r["image_id"] for r in merged_rows if r["split"] == "train"],
        "val": [r["image_id"] for r in merged_rows if r["split"] == "val"],
        "test": [r["image_id"] for r in merged_rows if r["split"] == "test"],
    }
    return all_rows, split_dict


def write_manifests(manifests: dict[str, list[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in manifests.items():
        save_jsonl(out_dir / f"{name}.jsonl", rows)
        pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)


def write_splits(split_dict: dict, out_path: Path) -> None:
    save_json(out_path, split_dict)


def summarize_manifest(rows: list[dict]) -> dict:
    split_counts = Counter(r["split"] for r in rows)
    label_format_counts = Counter(r["label_format"] for r in rows)
    class_counts = Counter()

    for row in rows:
        if row["label_format"] != "obb" or not row["labels_path"]:
            continue
        label_path = Path(row["labels_path"])
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            class_counts[int(float(parts[0]))] += 1

    return {
        "n_images": len(rows),
        "split_counts": dict(split_counts),
        "label_format_counts": dict(label_format_counts),
        "class_counts": dict(class_counts),
    }
