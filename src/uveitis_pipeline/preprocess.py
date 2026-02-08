from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import ensure_dir, read_image, save_json, write_image


class SamPromptMasker:
    def __init__(self, cfg: dict):
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        ckpt = Path(cfg["checkpoint"])
        if not ckpt.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")

        model_type = cfg.get("model_type", "vit_h")
        device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=str(ckpt))
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.cfg = cfg

    def _prompts(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        points = self.cfg.get(
            "points_norm",
            [
                [0.50, 0.50, 1],
                [0.35, 0.50, 1],
                [0.65, 0.50, 1],
                [0.50, 0.35, 1],
                [0.50, 0.65, 1],
                [0.03, 0.03, 0],
                [0.97, 0.03, 0],
                [0.03, 0.97, 0],
                [0.97, 0.97, 0],
                [0.50, 0.02, 0],
                [0.50, 0.98, 0],
            ],
        )
        coords = np.array([[float(x) * w, float(y) * h] for x, y, _ in points], dtype=np.float32)
        labels = np.array([int(l) for _, _, l in points], dtype=np.int32)
        return coords, labels

    @staticmethod
    def _fill_largest_contour(mask: np.ndarray) -> np.ndarray:
        m = (mask > 0).astype(np.uint8) * 255
        if not np.any(m):
            return np.zeros_like(m, dtype=np.uint8)
        res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]
        if not contours:
            return np.zeros_like(m, dtype=np.uint8)
        c = max(contours, key=cv2.contourArea)
        out = np.zeros_like(m, dtype=np.uint8)
        cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
        return (out > 0).astype(np.uint8)

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        open_k = int(self.cfg.get("open_kernel", 7))
        close_k = int(self.cfg.get("close_kernel", 19))
        if open_k > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if close_k > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        # Fill holes by taking everything inside the largest outline.
        return self._fill_largest_contour(mask)

    def _score(self, mask: np.ndarray, sam_score: float) -> float:
        h, w = mask.shape
        area = float(mask.mean())
        min_area = float(self.cfg.get("min_area_ratio", 0.10))
        max_area = float(self.cfg.get("max_area_ratio", 0.98))
        if area < min_area or area > max_area:
            return -1e9

        border_touch = float(
            (mask[0, :].mean() + mask[-1, :].mean() + mask[:, 0].mean() + mask[:, -1].mean()) / 4.0
        )
        if border_touch > float(self.cfg.get("max_border_touch_ratio", 0.55)):
            return -1e9

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return -1e9
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        bbox_fill = float(mask.sum()) / max(1.0, float((x1 - x0 + 1) * (y1 - y0 + 1)))
        center_hit = 1.0 if mask[h // 2, w // 2] > 0 else 0.0

        return float(sam_score) + 0.5 * area + 0.4 * bbox_fill + 0.25 * center_hit - 0.8 * border_touch

    def predict(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        self.predictor.set_image(image)
        point_coords, point_labels = self._prompts(h, w)
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=bool(self.cfg.get("multimask_output", True)),
        )

        best_score = -1e9
        best_mask = None
        for mask, sam_score in zip(masks, scores):
            clean = self._clean(mask.astype(np.uint8))
            score = self._score(clean, float(sam_score))
            if score > best_score:
                best_score = score
                best_mask = clean

        if best_mask is None:
            return np.zeros((h, w), dtype=np.uint8)
        return (best_mask > 0).astype(np.uint8) * 255


class Sam2PromptMasker:
    def __init__(self, cfg: dict):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        ckpt = Path(cfg["checkpoint"])
        if not ckpt.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt}")

        model_cfg = cfg.get("model_cfg", "configs/sam2.1/sam2.1_hiera_b+.yaml")
        device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build(model_cfg, ckpt, device=device, apply_postprocessing=bool(cfg.get("apply_postprocessing", True)))
        self.predictor = SAM2ImagePredictor(self.model)
        self.cfg = cfg
        self.device = str(device)

    @staticmethod
    def _build(model_cfg: str, ckpt: Path, device: str, apply_postprocessing: bool):
        from sam2.build_sam import build_sam2 as build_pkg

        return build_pkg(config_file=model_cfg, ckpt_path=str(ckpt), device=device, apply_postprocessing=apply_postprocessing)

    def _prompts(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        points = self.cfg.get(
            "points_norm",
            [
                [0.50, 0.50, 1],
                [0.35, 0.50, 1],
                [0.65, 0.50, 1],
                [0.50, 0.35, 1],
                [0.50, 0.65, 1],
                [0.03, 0.03, 0],
                [0.97, 0.03, 0],
                [0.03, 0.97, 0],
                [0.97, 0.97, 0],
                [0.50, 0.02, 0],
                [0.50, 0.98, 0],
            ],
        )
        coords = np.array([[float(x) * w, float(y) * h] for x, y, _ in points], dtype=np.float32)
        labels = np.array([int(l) for _, _, l in points], dtype=np.int32)
        return coords, labels

    @staticmethod
    def _fill_largest_contour(mask: np.ndarray) -> np.ndarray:
        m = (mask > 0).astype(np.uint8) * 255
        if not np.any(m):
            return np.zeros_like(m, dtype=np.uint8)
        res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]
        if not contours:
            return np.zeros_like(m, dtype=np.uint8)
        c = max(contours, key=cv2.contourArea)
        out = np.zeros_like(m, dtype=np.uint8)
        cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
        return (out > 0).astype(np.uint8)

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        open_k = int(self.cfg.get("open_kernel", 7))
        close_k = int(self.cfg.get("close_kernel", 19))
        if open_k > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if close_k > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        # Fill holes by taking everything inside the largest outline.
        return self._fill_largest_contour(mask)

    def _score(self, mask: np.ndarray, sam_score: float) -> float:
        h, w = mask.shape
        area = float(mask.mean())
        min_area = float(self.cfg.get("min_area_ratio", 0.10))
        max_area = float(self.cfg.get("max_area_ratio", 0.98))
        if area < min_area or area > max_area:
            return -1e9

        border_touch = float(
            (mask[0, :].mean() + mask[-1, :].mean() + mask[:, 0].mean() + mask[:, -1].mean()) / 4.0
        )
        if border_touch > float(self.cfg.get("max_border_touch_ratio", 0.55)):
            return -1e9

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return -1e9
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        bbox_fill = float(mask.sum()) / max(1.0, float((x1 - x0 + 1) * (y1 - y0 + 1)))
        center_hit = 1.0 if mask[h // 2, w // 2] > 0 else 0.0

        return float(sam_score) + 0.5 * area + 0.4 * bbox_fill + 0.25 * center_hit - 0.8 * border_touch

    def predict(self, image: np.ndarray) -> np.ndarray:
        import torch

        h, w = image.shape[:2]
        self.predictor.set_image(image)
        point_coords, point_labels = self._prompts(h, w)

        use_amp = self.device.startswith("cuda") and torch.cuda.is_available()
        if use_amp:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=bool(self.cfg.get("multimask_output", True)),
                )
        else:
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=bool(self.cfg.get("multimask_output", True)),
                )

        best_score = -1e9
        best_mask = None
        for mask, sam_score in zip(masks, scores):
            clean = self._clean(mask.astype(np.uint8))
            score = self._score(clean, float(sam_score))
            if score > best_score:
                best_score = score
                best_mask = clean

        if best_mask is None:
            return np.zeros((h, w), dtype=np.uint8)
        return (best_mask > 0).astype(np.uint8) * 255


def _compute_threshold_mask(image: np.ndarray, cfg: dict) -> np.ndarray:
    max_side = int(cfg.get("downsample_max_side", 768))
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    _, t_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sat_thr = int(cfg.get("sat_threshold", 18))
    mask = ((sat > sat_thr) | (t_gray > 0)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros((h, w), dtype=np.uint8)

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def _safe_erode(mask: np.ndarray, erode_px: int) -> np.ndarray:
    if erode_px <= 0:
        return mask
    k = int(erode_px) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded if np.any(eroded > 0) else mask


def compute_roi_mask(
    image: np.ndarray,
    cfg: dict,
    dataset: str = "",
    sam_masker: SamPromptMasker | None = None,
    sam2_masker: Sam2PromptMasker | None = None,
) -> np.ndarray:
    method_by_dataset = cfg.get("method_by_dataset", {})
    method = method_by_dataset.get(dataset, cfg.get("method", "threshold"))

    if method == "sam_prompted" and sam_masker is not None:
        mask = sam_masker.predict(image)
        if np.any(mask > 0):
            return mask

    if method == "sam2_prompted" and sam2_masker is not None:
        max_side = int(cfg.get("downsample_max_side", 768))
        h, w = image.shape[:2]
        scale = min(1.0, max_side / max(h, w)) if max_side > 0 else 1.0
        small = (
            cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else image
        )
        mask = sam2_masker.predict(small)
        if scale < 1.0:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if np.any(mask > 0):
            return mask

    return _compute_threshold_mask(image, cfg)


def crop_to_roi(image: np.ndarray, mask: np.ndarray, pad_px: int = 12) -> tuple[np.ndarray, dict]:
    ys, xs = np.where(mask > 0)
    h, w = image.shape[:2]
    if len(xs) == 0 or len(ys) == 0:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0 = max(0, int(xs.min()) - pad_px)
        y0 = max(0, int(ys.min()) - pad_px)
        x1 = min(w, int(xs.max()) + pad_px + 1)
        y1 = min(h, int(ys.max()) + pad_px + 1)
    crop = image[y0:y1, x0:x1]
    meta = {
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "orig_size": [int(w), int(h)],
        "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
    }
    return crop, meta


def normalize_color(
    image: np.ndarray,
    stats_mask: np.ndarray,
    method: str,
    out_mask: np.ndarray | None = None,
    ref: dict | None = None,
) -> tuple[np.ndarray, dict]:
    stats = stats_mask > 0
    # Exclude pure black pixels from stat estimation (regular fundus borders, masked-out regions).
    stats &= (image.astype(np.int32).sum(axis=2) > 0)
    out_apply = (out_mask > 0) if out_mask is not None else stats

    out = image.astype(np.float32).copy()
    roi_px = out[stats]
    if roi_px.size == 0:
        return image.copy(), {"method": method, "mean": [0, 0, 0], "std": [0, 0, 0]}

    mean = roi_px.mean(axis=0)
    std = roi_px.std(axis=0) + 1e-6

    if method == "zscore_rgb":
        norm = (out - mean[None, None, :]) / std[None, None, :]
        norm = np.clip(norm, -2.5, 2.5)
        out = ((norm + 2.5) / 5.0) * 255.0
    elif method == "grayworld":
        target = float(mean.mean())
        gains = target / std.clip(min=1e-6)
        out = out * gains[None, None, :]
    elif method == "clahe_luminance":
        lab = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if out_mask is None or not np.any(out_apply):
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        else:
            ys, xs = np.where(out_apply)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            sub = lab[y0:y1, x0:x1, 0]
            sub_mask = out_apply[y0:y1, x0:x1]
            # Keep the CLAHE histogram focused on ROI pixels.
            fill = int(np.median(sub[sub_mask])) if np.any(sub_mask) else int(np.median(sub))
            tmp = sub.copy()
            tmp[~sub_mask] = fill
            tmp = clahe.apply(tmp)
            sub[sub_mask] = tmp[sub_mask]
            lab[y0:y1, x0:x1, 0] = sub
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    elif method == "reinhard_lab_ref":
        if not ref or "lab_mean" not in ref or "lab_std" not in ref:
            raise ValueError("reinhard_lab_ref requires ref={'lab_mean': [...], 'lab_std': [...]} loaded from stats file")
        tgt_mean = np.array(ref["lab_mean"], dtype=np.float32).reshape(1, 1, 3)
        tgt_std = np.array(ref["lab_std"], dtype=np.float32).reshape(1, 1, 3)

        lab = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_roi = lab[stats]
        src_mean = lab_roi.mean(axis=0).astype(np.float32)
        src_std = (lab_roi.std(axis=0) + 1e-6).astype(np.float32)
        src_std = np.maximum(src_std, 1.0)

        lab = (lab - src_mean.reshape(1, 1, 3)) / src_std.reshape(1, 1, 3)
        lab = lab * tgt_std + tgt_mean
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    out[~out_apply] = 0.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    meta = {
        "method": method,
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
    }
    if method == "reinhard_lab_ref":
        meta["ref_lab_mean"] = [float(x) for x in ref["lab_mean"]]
        meta["ref_lab_std"] = [float(x) for x in ref["lab_std"]]
    return out, meta


def resize_global(image: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    side = max(h, w)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + h, pad_x : pad_x + w] = image
    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    meta = {
        "pad_xy": [int(pad_x), int(pad_y)],
        "padded_side": int(side),
        "scale": float(size / side),
        "orig_size": [int(w), int(h)],
    }
    return resized, meta


def tile(image: np.ndarray, tile_size: int, overlap: float) -> tuple[list[np.ndarray], list[dict]]:
    h, w = image.shape[:2]
    stride = max(1, int(tile_size * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    if not xs or xs[-1] != w - tile_size:
        xs.append(max(0, w - tile_size))
    if not ys or ys[-1] != h - tile_size:
        ys.append(max(0, h - tile_size))

    tiles: list[np.ndarray] = []
    metas: list[dict] = []
    idx = 0
    for y0 in ys:
        for x0 in xs:
            x1 = min(w, x0 + tile_size)
            y1 = min(h, y0 + tile_size)
            tile_img = image[y0:y1, x0:x1]
            if tile_img.shape[0] != tile_size or tile_img.shape[1] != tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[: tile_img.shape[0], : tile_img.shape[1]] = tile_img
                tile_img = padded
            tiles.append(tile_img)
            metas.append(
                {
                    "tile_id": f"tile_{idx:03d}",
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(min(w, x0 + tile_size)),
                    "y1": int(min(h, y0 + tile_size)),
                    "scale": 1.0,
                }
            )
            idx += 1
    return tiles, metas


def reconstruct_from_tiles(tiles: list[np.ndarray], metas: list[dict], out_size: tuple[int, int]) -> np.ndarray:
    h, w = out_size
    acc = np.zeros((h, w, 3), dtype=np.float32)
    cnt = np.zeros((h, w, 1), dtype=np.float32)
    for tile_img, meta in zip(tiles, metas):
        x0, y0, x1, y1 = meta["x0"], meta["y0"], meta["x1"], meta["y1"]
        view = tile_img[: y1 - y0, : x1 - x0].astype(np.float32)
        acc[y0:y1, x0:x1] += view
        cnt[y0:y1, x0:x1] += 1
    cnt[cnt == 0] = 1
    return (acc / cnt).astype(np.uint8)


def process_manifest(manifest_rows: list[dict], cfg: dict, out_root: Path) -> dict:
    roi_dir = ensure_dir(out_root / "roi_masks")
    crop_dir = ensure_dir(out_root / "crops")
    crop_meta_dir = ensure_dir(out_root / "crop_meta")
    norm_dir = ensure_dir(out_root / "norm")
    norm_meta_dir = ensure_dir(out_root / "norm_meta")
    global_dir = ensure_dir(out_root / "global_1024")
    tiles_dir = ensure_dir(out_root / "tiles")
    tiles_meta_dir = ensure_dir(out_root / "tiles_meta")

    roi_cfg = cfg["roi"]
    sam_cfg = roi_cfg.get("sam", {})
    sam2_cfg = roi_cfg.get("sam2", {})
    use_sam = (
        roi_cfg.get("method") == "sam_prompted"
        or any(v == "sam_prompted" for v in roi_cfg.get("method_by_dataset", {}).values())
    )
    use_sam2 = (
        roi_cfg.get("method") == "sam2_prompted"
        or any(v == "sam2_prompted" for v in roi_cfg.get("method_by_dataset", {}).values())
    )

    sam_masker = None
    sam_failures = 0
    if use_sam:
        try:
            sam_masker = SamPromptMasker(sam_cfg)
        except Exception:
            if not bool(sam_cfg.get("fallback_to_threshold", True)):
                raise

    sam2_masker = None
    sam2_failures = 0
    if use_sam2:
        try:
            sam2_masker = Sam2PromptMasker(sam2_cfg)
        except Exception:
            if not bool(sam2_cfg.get("fallback_to_threshold", True)):
                raise

    ref_stats = None
    norm_cfg = cfg.get("normalize", {})
    norm_method = str(norm_cfg.get("method", "zscore_rgb"))
    if norm_method == "reinhard_lab_ref":
        ref_path = Path(norm_cfg.get("ref", {}).get("stats_path", ""))
        if not ref_path:
            raise ValueError("normalize.ref.stats_path is required for normalize.method=reinhard_lab_ref")
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Missing color reference stats: {ref_path}. Run scripts/build_regular_fundus_color_ref.py first."
            )
        import json

        ref_stats = json.loads(ref_path.read_text(encoding="utf-8"))

    fail_small = 0
    fail_border = 0

    for row in manifest_rows:
        image_id = row["image_id"].replace("::", "__")
        image = read_image(row["filepath"])

        mask = compute_roi_mask(
            image,
            roi_cfg,
            dataset=row.get("dataset", ""),
            sam_masker=sam_masker,
            sam2_masker=sam2_masker,
        )
        if use_sam and row.get("dataset") == "uwf700" and sam_masker is None:
            sam_failures += 1
        if use_sam2 and row.get("dataset") == "uwf700" and sam2_masker is None:
            sam2_failures += 1
        write_image(roi_dir / f"{image_id}.png", np.repeat(mask[:, :, None], 3, axis=2))

        crop, crop_meta = crop_to_roi(image, mask, int(roi_cfg.get("crop_pad_px", 12)))
        write_image(crop_dir / f"{image_id}.png", crop)
        save_json(crop_meta_dir / f"{image_id}.json", crop_meta)

        x0, y0, x1, y1 = crop_meta["bbox_xyxy"]
        roi_crop = mask[y0:y1, x0:x1]
        stats_mask = _safe_erode(roi_crop, int(cfg["normalize"].get("stats_erode_px", 4)))
        norm, norm_meta = normalize_color(crop, stats_mask, norm_method, out_mask=roi_crop, ref=ref_stats)
        write_image(norm_dir / f"{image_id}.png", norm)
        save_json(norm_meta_dir / f"{image_id}.json", norm_meta)

        global_img, global_meta = resize_global(norm, int(cfg["resize"]["global_size"]))
        write_image(global_dir / f"{image_id}.png", global_img)

        tiles, metas = tile(global_img, int(cfg["tiling"]["tile_size"]), float(cfg["tiling"]["overlap"]))
        tile_img_dir = ensure_dir(tiles_dir / image_id)
        for tile_img, meta in zip(tiles, metas):
            write_image(tile_img_dir / f"{meta['tile_id']}.png", tile_img)

        save_json(
            tiles_meta_dir / f"{image_id}.json",
            {
                "image_id": row["image_id"],
                "global_size": [int(global_img.shape[1]), int(global_img.shape[0])],
                "global_meta": global_meta,
                "tiles": metas,
            },
        )

        roi_area_ratio = float((mask > 0).mean())
        border_touch = x0 == 0 or y0 == 0 or x1 == image.shape[1] or y1 == image.shape[0]
        if roi_area_ratio < float(cfg["verify"]["min_roi_area_ratio"]):
            fail_small += 1
        if border_touch:
            fail_border += 1

    n = max(1, len(manifest_rows))
    return {
        "num_images": len(manifest_rows),
        "roi_small_rate": fail_small / n,
        "roi_border_touch_rate": fail_border / n,
        "sam_init_failed": int(sam_masker is None and use_sam),
        "sam_fallback_images": sam_failures,
        "sam2_init_failed": int(sam2_masker is None and use_sam2),
        "sam2_fallback_images": sam2_failures,
    }
