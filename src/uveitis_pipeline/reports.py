"""Dataset and preprocessing reporting helpers for roadmap checkpoints."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .common import ensure_dir, read_image, read_jsonl, save_json, write_image


def _global_path(preproc_root: Path, image_key: str) -> Path:
    """Resolve preprocessed global image path."""
    p = preproc_root / "global" / f"{image_key}.png"
    if p.exists():
        return p
    return preproc_root / "global_1024" / f"{image_key}.png"


def report_dataset(manifests: list[str], label_indexes: list[str], out: str) -> str:
    """Write dataset and label distribution report (native labels or COCO)."""
    rows: list[dict] = []
    for path in manifests:
        rows.extend(read_jsonl(path))

    by_dataset = Counter(r.get("dataset", "unknown") for r in rows)
    by_split = Counter(r.get("split", "") for r in rows)
    by_format = Counter(r.get("label_format", "") for r in rows)

    label_stats = []
    for path in label_indexes:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix == ".jsonl":
            recs = read_jsonl(p)
            label_stats.append(
                {
                    "path": p.as_posix(),
                    "kind": "native_index",
                    "num_records": len(recs),
                    "num_objects": int(sum(int(r.get("num_objects", 0)) for r in recs)),
                    "non_empty_records": int(sum(1 for r in recs if int(r.get("num_objects", 0)) > 0)),
                }
            )
            continue

        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "images" in data and "annotations" in data:
            label_stats.append(
                {
                    "path": p.as_posix(),
                    "kind": "coco",
                    "num_images": len(data["images"]),
                    "num_annotations": len(data["annotations"]),
                }
            )

    out_path = Path(out)
    ensure_dir(out_path.parent)

    payload = {
        "num_manifest_rows": len(rows),
        "dataset_counts": dict(by_dataset),
        "split_counts": dict(by_split),
        "label_format_counts": dict(by_format),
        "label_indexes": label_stats,
    }
    save_json(out_path.with_suffix(".json"), payload)

    lines = ["# Dataset Report", "", f"Rows: {len(rows)}", "", "## Datasets"]
    for k, v in sorted(by_dataset.items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Splits"]
    for k, v in sorted(by_split.items()):
        lines.append(f"- {k or 'unset'}: {v}")
    lines += ["", "## Label Formats"]
    for k, v in sorted(by_format.items()):
        lines.append(f"- {k or 'unset'}: {v}")
    if label_stats:
        lines += ["", "## Label Indexes"]
        for row in label_stats:
            lines.append(f"- {row}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path.as_posix()


def _build_triptych(raw: np.ndarray, mask_rgb: np.ndarray, norm: np.ndarray, max_side: int = 560) -> np.ndarray:
    """Compose a raw/mask/norm preview strip."""

    def _preview(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(1.0, float(max_side) / max(h, w))
        if scale >= 1.0:
            return img
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _pad_h(img: np.ndarray, h: int) -> np.ndarray:
        if img.shape[0] == h:
            return img
        if img.shape[0] > h:
            return img[:h]
        pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
        return np.concatenate([img, pad], axis=0)

    a, b, c = _preview(raw), _preview(mask_rgb), _preview(norm)
    hh = max(a.shape[0], b.shape[0], c.shape[0])
    return np.concatenate([_pad_h(a, hh), _pad_h(b, hh), _pad_h(c, hh)], axis=1)


def report_preproc(manifest: str, preproc_root: str, out_dir: str, sample_n: int = 24) -> str:
    """Write preprocessing coverage report and sample visual checks."""
    rows = read_jsonl(manifest)
    preproc = Path(preproc_root)
    out = ensure_dir(Path(out_dir))
    vis_dir = ensure_dir(out / "triptychs")

    present = defaultdict(int)
    miss = defaultdict(int)
    available: list[dict] = []

    for row in rows:
        key = row["image_id"].replace("::", "__")
        paths = {
            "roi": preproc / "roi_masks" / f"{key}.png",
            "crop": preproc / "crops" / f"{key}.png",
            "norm": preproc / "norm" / f"{key}.png",
            "global": _global_path(preproc, key),
            "tiles_meta": preproc / "tiles_meta" / f"{key}.json",
        }
        ok = True
        for name, p in paths.items():
            if p.exists():
                present[name] += 1
            else:
                miss[name] += 1
                ok = False
        if ok:
            available.append(row)

    sample = available[: max(0, int(sample_n))]
    for row in sample:
        key = row["image_id"].replace("::", "__")
        raw = read_image(row["filepath"])
        mask_rgb = read_image(preproc / "roi_masks" / f"{key}.png")
        norm = read_image(preproc / "norm" / f"{key}.png")
        tri = _build_triptych(raw, mask_rgb, norm)
        write_image(vis_dir / f"{key}.png", tri)

    metrics_path = preproc / "verify" / "preprocess_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    summary = {
        "manifest": manifest,
        "preproc_root": preproc.as_posix(),
        "num_rows": len(rows),
        "num_fully_available": len(available),
        "present_counts": dict(present),
        "missing_counts": dict(miss),
        "metrics": metrics,
        "triptych_dir": vis_dir.as_posix(),
    }
    save_json(out / "preproc_report.json", summary)

    md = ["# Preprocess Report", "", f"Rows: {len(rows)}", f"Fully available: {len(available)}", "", "## Present"]
    md.extend([f"- {k}: {v}" for k, v in sorted(present.items())])
    md += ["", "## Missing"]
    md.extend([f"- {k}: {v}" for k, v in sorted(miss.items())])
    if metrics:
        md += ["", "## Metrics", f"```json\n{json.dumps(metrics, indent=2)}\n```"]
    (out / "preproc_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return (out / "preproc_report.md").as_posix()
