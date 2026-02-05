#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import random
import shutil
from collections import defaultdict

import cv2
import numpy as np

from uveitis_pipeline.common import ensure_dir, load_yaml, read_image, write_image
from uveitis_pipeline.preprocess import Sam2PromptMasker, _compute_threshold_mask


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _preview(image: np.ndarray, max_side: int = 420) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _resize_max_side(image: np.ndarray, max_side: int, interpolation: int) -> np.ndarray:
    if max_side <= 0:
        return image
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=interpolation)


def _pad_to_height(image: np.ndarray, h: int) -> np.ndarray:
    if image.shape[0] == h:
        return image
    if image.shape[0] > h:
        return image[:h]
    pad = np.zeros((h - image.shape[0], image.shape[1], 3), dtype=image.dtype)
    return np.concatenate([image, pad], axis=0)


def _overlay_boundary(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    edges = cv2.Canny(m, 80, 160)
    out = image.copy()
    out[edges > 0] = np.array(color, dtype=np.uint8)
    return out


def _fill_largest_contour(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8) * 255
    if not np.any(m):
        return np.zeros_like(m, dtype=np.uint8)
    res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    if not contours:
        return np.zeros_like(m, dtype=np.uint8)
    c = max(contours, key=cv2.contourArea)
    out = np.zeros_like(m, dtype=np.uint8)
    cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
    return (out > 0).astype(np.uint8) * 255


def _clean_mask(mask01: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    mask = (mask01 > 0).astype(np.uint8) * 255
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return _fill_largest_contour(mask)


def _score_mask(mask_u8: np.ndarray, sam_score: float, cfg: dict) -> float:
    m = (mask_u8 > 0).astype(np.uint8)
    h, w = m.shape
    area = float(m.mean())
    if area < float(cfg.get("min_area_ratio", 0.10)) or area > float(cfg.get("max_area_ratio", 0.98)):
        return -1e9

    border_touch = float((m[0, :].mean() + m[-1, :].mean() + m[:, 0].mean() + m[:, -1].mean()) / 4.0)
    if border_touch > float(cfg.get("max_border_touch_ratio", 0.55)):
        return -1e9

    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return -1e9
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bbox_fill = float(m.sum()) / max(1.0, float((x1 - x0 + 1) * (y1 - y0 + 1)))
    center_hit = 1.0 if m[h // 2, w // 2] > 0 else 0.0
    return float(sam_score) + 0.5 * area + 0.4 * bbox_fill + 0.25 * center_hit - 0.8 * border_touch


def _metrics(mask_u8: np.ndarray) -> dict:
    m = (mask_u8 > 0).astype(np.uint8)
    h, w = m.shape
    area = float(m.mean())
    border_touch = float((m[0, :].mean() + m[-1, :].mean() + m[:, 0].mean() + m[:, -1].mean()) / 4.0)
    ring = max(3, int(0.04 * min(h, w)))
    ring_mask = np.zeros_like(m, dtype=bool)
    ring_mask[:ring, :] = True
    ring_mask[-ring:, :] = True
    ring_mask[:, :ring] = True
    ring_mask[:, -ring:] = True
    ring_frac = float(m[ring_mask].mean())
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return {"area": 0.0, "border_touch": 0.0, "ring_frac": 0.0, "bbox_fill": 0.0}
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bbox_fill = float(m.sum()) / max(1.0, float((x1 - x0 + 1) * (y1 - y0 + 1)))
    return {"area": area, "border_touch": border_touch, "ring_frac": ring_frac, "bbox_fill": bbox_fill}


def _prompts_from_norm(points_norm: list[list[float]], h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.array([[float(x) * w, float(y) * h] for x, y, _ in points_norm], dtype=np.float32)
    labels = np.array([int(l) for _, _, l in points_norm], dtype=np.int32)
    return coords, labels


def _run_sam2_candidates(masker: Sam2PromptMasker, image: np.ndarray, points_norm: list[list[float]]) -> list[tuple[np.ndarray, float]]:
    import torch

    h, w = image.shape[:2]
    coords, labels = _prompts_from_norm(points_norm, h, w)

    use_amp = str(masker.device).startswith("cuda") and torch.cuda.is_available()
    if use_amp:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = masker.predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=bool(masker.cfg.get("multimask_output", True)),
            )
    else:
        with torch.inference_mode():
            masks, scores, _ = masker.predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=bool(masker.cfg.get("multimask_output", True)),
            )
    return [(m.astype(np.uint8) * 255, float(s)) for m, s in zip(masks, scores)]


def _set_image(masker: Sam2PromptMasker, image: np.ndarray) -> None:
    # SAM2 does an expensive image embedding on set_image; call it once per image and reuse.
    masker.predictor.set_image(image)


def _best_from_candidates(cands: list[tuple[np.ndarray, float]], cfg: dict) -> tuple[np.ndarray, float]:
    open_k = int(cfg.get("open_kernel", 7))
    close_k = int(cfg.get("close_kernel", 19))
    best_s = -1e9
    best = None
    for m, s in cands:
        clean = _clean_mask(m, open_k, close_k)
        score = _score_mask(clean, float(s), cfg)
        if score > best_s:
            best_s = score
            best = clean
    if best is None:
        h, w = cands[0][0].shape if cands else (1, 1)
        return np.zeros((h, w), dtype=np.uint8), -1e9
    return best, float(best_s)


def _bbox_from_mask(mask_u8: np.ndarray, pad: int, w: int, h: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y1 = min(h, int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def _threshold_guided_points(image: np.ndarray, thr_mask: np.ndarray) -> list[list[float]]:
    h, w = image.shape[:2]
    m = thr_mask > 0
    ys, xs = np.where(m)
    if len(xs) == 0:
        return []
    cx, cy = float(xs.mean()), float(ys.mean())
    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    pts = [
        [cx / w, cy / h, 1],
        [(x0 + 0.25 * (x1 - x0)) / w, cy / h, 1],
        [(x0 + 0.75 * (x1 - x0)) / w, cy / h, 1],
        [cx / w, (y0 + 0.25 * (y1 - y0)) / h, 1],
        [cx / w, (y0 + 0.75 * (y1 - y0)) / h, 1],
        [0.03, 0.03, 0],
        [0.97, 0.03, 0],
        [0.03, 0.97, 0],
        [0.97, 0.97, 0],
        [0.50, 0.02, 0],
        [0.50, 0.98, 0],
        [0.02, 0.50, 0],
        [0.98, 0.50, 0],
    ]
    return pts


def _threshold_guided_points_with_negatives(image: np.ndarray, thr_mask: np.ndarray, n_neg: int = 8) -> list[list[float]]:
    h, w = image.shape[:2]
    m = (thr_mask > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return []

    cx, cy = float(xs.mean()), float(ys.mean())
    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    pts = [
        [cx / w, cy / h, 1],
        [(x0 + 0.10 * (x1 - x0)) / w, cy / h, 1],
        [(x0 + 0.90 * (x1 - x0)) / w, cy / h, 1],
        [cx / w, (y0 + 0.10 * (y1 - y0)) / h, 1],
        [cx / w, (y0 + 0.90 * (y1 - y0)) / h, 1],
        [0.03, 0.03, 0],
        [0.97, 0.03, 0],
        [0.03, 0.97, 0],
        [0.97, 0.97, 0],
        [0.50, 0.02, 0],
        [0.50, 0.98, 0],
        [0.02, 0.50, 0],
        [0.98, 0.50, 0],
    ]

    # Sample extra negative points from outside threshold mask, biased to image borders where machinery often lives.
    rng = np.random.default_rng(42)
    inv = m == 0
    ring = max(3, int(0.12 * min(h, w)))
    border = np.zeros((h, w), dtype=bool)
    border[:ring, :] = True
    border[-ring:, :] = True
    border[:, :ring] = True
    border[:, -ring:] = True
    cand = np.where(inv & border)
    if cand[0].size == 0:
        cand = np.where(inv)
    if cand[0].size > 0 and n_neg > 0:
        take = min(int(n_neg), int(cand[0].size))
        idx = rng.choice(int(cand[0].size), size=take, replace=False)
        ys2, xs2 = cand[0][idx], cand[1][idx]
        for y, x in zip(ys2.tolist(), xs2.tolist()):
            pts.append([float(x) / w, float(y) / h, 0])
    return pts


def _iou(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    a = a_u8 > 0
    b = b_u8 > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _recall(a_u8: np.ndarray, ref_u8: np.ndarray) -> float:
    a = a_u8 > 0
    r = ref_u8 > 0
    denom = float(r.sum())
    if denom <= 0:
        return 0.0
    return float(np.logical_and(a, r).sum()) / denom


def _leak(a_u8: np.ndarray, ref_u8: np.ndarray, dilate_px: int = 18) -> float:
    a = (a_u8 > 0).astype(np.uint8)
    r = (ref_u8 > 0).astype(np.uint8)
    if dilate_px > 0:
        k = int(dilate_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        r = cv2.dilate(r, kernel, iterations=1)
    denom = float(a.sum())
    if denom <= 0:
        return 0.0
    leak = float((a & (1 - r)).sum())
    return leak / denom


def _pick_diverse(images_root: Path, n: int, seed: int = 42) -> list[Path]:
    rng = random.Random(seed)
    buckets: dict[str, list[Path]] = {}
    for d in sorted(images_root.iterdir()):
        if not d.is_dir():
            continue
        imgs = [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]
        if imgs:
            buckets[d.name] = imgs
    keys = list(buckets.keys())
    if not keys:
        return []
    picked: list[Path] = []
    i = 0
    while len(picked) < n:
        k = keys[i % len(keys)]
        pool = buckets[k]
        if pool:
            picked.append(pool.pop(rng.randrange(len(pool))))
        i += 1
        if i > 100000:
            break
    return picked[:n]


def main() -> None:
    p = argparse.ArgumentParser(description="Compare SAM2 fundus masking strategies (no GT; heuristic scoring).")
    p.add_argument("--config", default="configs/stage0_preprocess.yaml")
    p.add_argument("--images-root", default="datasets/uwf-700/Images")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default="eval/sam2_mask_compare_uwf20")
    p.add_argument("--save-max-side", type=int, default=1024, help="Downscale saved overlays/masks for speed (0 keeps full res).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    cfg = load_yaml(args.config)
    roi_cfg = cfg["roi"]
    sam2_cfg = roi_cfg.get("sam2", {})
    ds_max_side = int(roi_cfg.get("downsample_max_side", 768))

    masker = Sam2PromptMasker(sam2_cfg)
    base_points = list(sam2_cfg.get("points_norm", []))
    if not base_points:
        raise ValueError("roi.sam2.points_norm is empty")

    # 1) Ensemble: run separate prompt templates, pool all candidates, pick best by scoring.
    ensemble_points = [
        base_points,
        base_points
        + [
            [0.25, 0.02, 0],
            [0.75, 0.02, 0],
            [0.25, 0.98, 0],
            [0.75, 0.98, 0],
            [0.02, 0.50, 0],
            [0.98, 0.50, 0],
        ],
        [
            [0.50, 0.50, 1],
            [0.38, 0.50, 1],
            [0.62, 0.50, 1],
            [0.50, 0.38, 1],
            [0.50, 0.62, 1],
            [0.42, 0.42, 1],
            [0.58, 0.42, 1],
            [0.42, 0.58, 1],
            [0.58, 0.58, 1],
            [0.03, 0.03, 0],
            [0.97, 0.03, 0],
            [0.03, 0.97, 0],
            [0.97, 0.97, 0],
            [0.50, 0.02, 0],
            [0.50, 0.98, 0],
        ],
    ]

    # baseline: current prompts, best-by-score
    # ensemble_prompts: multiple separate predictions, pooled and scored
    # threshold_prior_select: choose candidate that covers threshold fundus while minimizing leakage
    # threshold_guided_negatives: threshold-derived positive prompts + extra negatives near borders
    methods = ["baseline", "ensemble_prompts", "threshold_prior_select", "threshold_guided_negatives"]
    for m in methods:
        for sub in ["masks", "overlays", "masked", "compare"]:
            ensure_dir(out_dir / m / sub)
    ensure_dir(out_dir / "comparisons")

    picked = _pick_diverse(Path(args.images_root), int(args.n), seed=42)
    if not picked:
        raise RuntimeError(f"No images found under {args.images_root}")

    all_rows = []
    agg = defaultdict(list)

    for img_path in picked:
        if len(all_rows) and len(all_rows) % 5 == 0:
            print(json.dumps({"processed": len(all_rows), "total": len(picked)}))
        image = read_image(img_path)
        h, w = image.shape[:2]
        key = f"{img_path.parent.name}__{img_path.stem}"

        # Mirror preprocessing behavior: run SAM2 at a capped resolution for speed, then upscale.
        scale = min(1.0, float(ds_max_side) / max(h, w)) if ds_max_side > 0 else 1.0
        small = (
            cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else image
        )
        hs, ws = small.shape[:2]
        thr = _compute_threshold_mask(small, roi_cfg)
        thr_full = cv2.resize(thr, (w, h), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else thr

        _set_image(masker, small)

        # baseline
        cands = _run_sam2_candidates(masker, small, base_points)
        m_base, s_base = _best_from_candidates(cands, sam2_cfg)

        # ensemble prompts
        pooled = []
        for pts in ensemble_points:
            pooled.extend(_run_sam2_candidates(masker, small, pts))
        m_ens, s_ens = _best_from_candidates(pooled, sam2_cfg)

        # threshold prior select: prefer masks that cover threshold fundus while not leaking outside it.
        # We score candidates using (fundus recall) - (leakage outside dilated fundus) with mild regularizers.
        def _prior_pick(candidates: list[tuple[np.ndarray, float]]) -> tuple[np.ndarray, float]:
            best = None
            best_s = -1e9
            for m_u8, sam_s in candidates:
                clean = _clean_mask(m_u8, int(sam2_cfg.get("open_kernel", 7)), int(sam2_cfg.get("close_kernel", 19)))
                if not np.any(clean > 0):
                    continue
                fundus_recall = _recall(clean, thr)
                leak = _leak(clean, thr, dilate_px=max(12, int(0.03 * min(hs, ws))))
                met = _metrics(clean)
                score = float(sam_s) + 1.7 * fundus_recall - 2.2 * leak - 0.7 * met["ring_frac"] + 0.3 * met["bbox_fill"]
                if score > best_s:
                    best_s = score
                    best = clean
            if best is None:
                return m_base, float(s_base)
            return best, float(best_s)

        m_prior, s_prior = _prior_pick(pooled if pooled else cands)

        # threshold guided negatives: use threshold-derived positive points + extra negatives near border.
        guided_pts = _threshold_guided_points_with_negatives(small, thr, n_neg=10)
        m_neg, s_neg = m_base, s_base
        if guided_pts:
            cands4 = _run_sam2_candidates(masker, small, guided_pts)
            m_neg, s_neg = _best_from_candidates(cands4, sam2_cfg)

        out_by_method = {
            "baseline": (m_base, s_base),
            "ensemble_prompts": (m_ens, s_ens),
            "threshold_prior_select": (m_prior, s_prior),
            "threshold_guided_negatives": (m_neg, s_neg),
        }

        # Save per-method artifacts + per-image comparison strips.
        strip = []
        for m in methods:
            mask_u8, sc = out_by_method[m]
            if scale < 1.0:
                mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = _overlay_boundary(image, mask_u8)
            masked = image.copy()
            masked[mask_u8 == 0] = 0
            save_overlay = _resize_max_side(overlay, int(args.save_max_side), cv2.INTER_AREA)
            save_mask = _resize_max_side(mask_u8, int(args.save_max_side), cv2.INTER_NEAREST)
            save_mask_rgb = np.repeat(save_mask[:, :, None], 3, axis=2)
            save_masked = _resize_max_side(masked, int(args.save_max_side), cv2.INTER_AREA)
            write_image(out_dir / m / "masks" / f"{key}.png", save_mask_rgb)
            write_image(out_dir / m / "overlays" / f"{key}.png", save_overlay)
            write_image(out_dir / m / "masked" / f"{key}.png", save_masked)

            pv = _preview(overlay)
            cv2.putText(
                pv,
                f"{m} score={sc:.3f}",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            strip.append(pv)

            met = _metrics(mask_u8)
            met["thr_iou"] = _iou(mask_u8, thr_full)
            met["thr_recall"] = _recall(mask_u8, thr_full)
            met["thr_leak"] = _leak(mask_u8, thr_full, dilate_px=max(12, int(0.03 * min(h, w))))
            agg[m].append(met)

        hh = max(x.shape[0] for x in strip)
        strip = [_pad_to_height(x, hh) for x in strip]
        comp = np.concatenate(strip, axis=1)
        write_image(out_dir / "comparisons" / f"{key}.png", comp)

        all_rows.append({"key": key, "path": str(img_path), "scores": {k: float(v[1]) for k, v in out_by_method.items()}})

    (out_dir / "selected_images.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    summary = {}
    for m in methods:
        mets = agg[m]
        if not mets:
            continue
        summary[m] = {k: float(np.mean([x[k] for x in mets])) for k in mets[0].keys()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"out": str(out_dir), "n": len(all_rows), "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
