from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def nonblack_mask(rgb: np.ndarray, thresh: int = 8) -> np.ndarray:
    """Pixels that are part of the (masked) fundus region, not the padded black background."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB image HxWx3")
    return (rgb.astype(np.int16).sum(axis=2) >= int(thresh)).astype(np.uint8)


@dataclass(frozen=True)
class VesselMaskParams:
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    blackhat_sizes: tuple[int, ...] = (11, 17, 23)
    thresh_quantile: float = 0.90
    open_ksize: int = 3


def vessel_mask(rgb: np.ndarray, nb: np.ndarray | None = None, p: VesselMaskParams = VesselMaskParams()) -> np.ndarray:
    """
    Fast, training-free vesselness proxy:
    - vessels are typically dark (esp. in green channel)
    - multi-scale blackhat highlights dark thin structures
    Returns a uint8 mask (0/1).
    """
    g = rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=float(p.clahe_clip), tileGridSize=(int(p.clahe_grid), int(p.clahe_grid)))
    g = clahe.apply(g)

    bh = np.zeros_like(g, dtype=np.uint8)
    for k in p.blackhat_sizes:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))
        bh = np.maximum(bh, cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, ker))

    bh_f = bh.astype(np.float32)
    if nb is not None:
        bh_f = bh_f * (nb.astype(np.float32) > 0)

    vals = bh_f[bh_f > 0]
    if vals.size < 64:
        return np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

    t = float(np.quantile(vals, float(p.thresh_quantile)))
    m = (bh_f >= t).astype(np.uint8)

    k = int(p.open_ksize)
    if k > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=1)
    if nb is not None:
        m = (m & (nb.astype(np.uint8) > 0)).astype(np.uint8)
    return m


def obb_norm_to_poly_px(obb_norm: list[float], w: int, h: int) -> np.ndarray:
    if len(obb_norm) != 8:
        raise ValueError("obb must have 8 floats (x1 y1 ... x4 y4) normalized")
    pts = np.array([(obb_norm[i] * w, obb_norm[i + 1] * h) for i in range(0, 8, 2)], dtype=np.float32)
    return pts


def bbox_to_poly_px(bbox_xywh: list[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], dtype=np.float32)


def point_in_any_poly(x: int, y: int, polys: list[np.ndarray]) -> bool:
    pt = (float(x), float(y))
    for poly in polys:
        if poly is None or len(poly) < 3:
            continue
        if cv2.pointPolygonTest(poly, pt, False) >= 0:
            return True
    return False


def extract_patch(rgb: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    hs = int(size) // 2
    h, w = rgb.shape[:2]
    x0, x1 = x - hs, x + hs
    y0, y1 = y - hs, y + hs
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - (w - 1))
    pad_b = max(0, y1 - (h - 1))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w - 1, x1)
    y1 = min(h - 1, y1)

    patch = rgb[y0 : y1 + 1, x0 : x1 + 1]
    if pad_l or pad_t or pad_r or pad_b:
        patch = cv2.copyMakeBorder(patch, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return patch


def mask_to_obbs(mask: np.ndarray, min_area: int = 40) -> list[tuple[np.ndarray, float]]:
    """
    Convert a binary mask to oriented boxes using minAreaRect.
    Returns [(poly4_px, score)] where score is component mean (assumes mask is 0/1).
    """
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out: list[tuple[np.ndarray, float]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if int(area) < int(min_area):
            continue
        comp = (labels == i).astype(np.uint8)
        ys, xs = np.where(comp > 0)
        if xs.size < 3:
            continue
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.float32)  # 4x2
        out.append((box, float(area)))
    out.sort(key=lambda t: t[1], reverse=True)
    return out

