"""Runtime inference service utilities for API and web UI."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch

from .common import load_yaml
from .preprocess import Sam2PromptMasker, _safe_erode, compute_roi_mask, crop_to_roi, normalize_color, resize_global
from .retfound_mask import RetFoundEncoder, RetFoundMaskModel, load_retfound_vit, load_retfound_weights


def _xyxy_iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between axis-aligned boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def _nms(preds: list[dict], iou_thr: float) -> list[dict]:
    """Apply per-class NMS."""
    out: list[dict] = []
    by_class: dict[int, list[dict]] = {}
    for p in preds:
        by_class.setdefault(int(p["class_id"]), []).append(p)
    for items in by_class.values():
        items = sorted(items, key=lambda x: float(x["score"]), reverse=True)
        keep: list[dict] = []
        while items:
            cur = items.pop(0)
            keep.append(cur)
            nxt = []
            for cand in items:
                if _xyxy_iou(cur["bbox_xyxy"], cand["bbox_xyxy"]) <= iou_thr:
                    nxt.append(cand)
            items = nxt
        out.extend(keep)
    return out


def _cap_per_class(preds: list[dict], max_per_class: int) -> list[dict]:
    """Keep top-K predictions per class."""
    k = int(max_per_class)
    if k <= 0:
        return preds
    out: list[dict] = []
    by_class: dict[int, list[dict]] = {}
    for p in preds:
        by_class.setdefault(int(p["class_id"]), []).append(p)
    for items in by_class.values():
        out.extend(sorted(items, key=lambda x: float(x["score"]), reverse=True)[:k])
    return out


def _extract_components(prob: np.ndarray, cls_id: int, cls_name: str, cfg: dict) -> list[dict]:
    """Convert a class probability map to polygon/box detections."""
    thr = float(cfg.get("threshold", 0.5))
    min_area = int(cfg.get("min_area_px", 16))
    simplify_eps = float(cfg.get("polygon_simplify_eps", 1.25))
    open_k = int(cfg.get("open_kernel", 0))
    close_k = int(cfg.get("close_kernel", 3))

    mask = (prob >= thr).astype(np.uint8)
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    h, w = mask.shape
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    out: list[dict] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(min_area):
            continue
        poly = cnt.reshape(-1, 2).astype(np.float32)
        if simplify_eps > 0:
            poly = cv2.approxPolyDP(poly, epsilon=simplify_eps, closed=True).reshape(-1, 2).astype(np.float32)
        if poly.shape[0] < 3:
            continue
        comp = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(comp, [np.round(poly).astype(np.int32)], 1)
        score = float(prob[comp > 0].mean()) if np.any(comp > 0) else float(prob[int(poly[0, 1]), int(poly[0, 0])])

        x1 = float(np.clip(poly[:, 0].min(), 0, w - 1))
        y1 = float(np.clip(poly[:, 1].min(), 0, h - 1))
        x2 = float(np.clip(poly[:, 0].max(), 0, w - 1))
        y2 = float(np.clip(poly[:, 1].max(), 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        rect = cv2.minAreaRect(poly)
        obb = cv2.boxPoints(rect).astype(np.float32)
        out.append(
            {
                "class_id": int(cls_id),
                "class_name": cls_name,
                "score": score,
                "bbox_xyxy": [x1, y1, x2, y2],
                "polygon": [float(np.clip(v / (w if i % 2 == 0 else h), 0.0, 1.0)) for i, v in enumerate([vv for xy in poly.tolist() for vv in xy])],
                "obb": [
                    float(np.clip(v / (w if i % 2 == 0 else h), 0.0, 1.0))
                    for i, v in enumerate([vv for xy in obb.tolist() for vv in xy])
                ],
            }
        )
    return out


def _encode_png_b64(image_rgb: np.ndarray) -> str:
    """Encode RGB image as base64 PNG string."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes to RGB."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _poly_norm_to_px(poly: list[float], w: int, h: int) -> np.ndarray:
    """Convert normalized polygon to pixel points."""
    pts = [[float(poly[i]) * w, float(poly[i + 1]) * h] for i in range(0, len(poly), 2)]
    return np.array(pts, dtype=np.float32)


def _draw_predictions(image_rgb: np.ndarray, preds: list[dict], color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw polygon detections on an RGB image."""
    out = image_rgb.copy()
    h, w = out.shape[:2]
    for p in preds:
        pts = _poly_norm_to_px(p.get("polygon", []), w, h)
        if pts.shape[0] < 3:
            continue
        pi = np.round(pts).astype(np.int32)
        cv2.polylines(out, [pi], True, color, 2, cv2.LINE_AA)
        x, y = int(np.min(pi[:, 0])), int(np.min(pi[:, 1]))
        txt = f"{p['class_name']}:{float(p.get('score', 0.0)):.2f}"
        cv2.putText(out, txt, (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def _convert_global_pred_to_original(pred: dict, crop_meta: dict, global_meta: dict, orig_w: int, orig_h: int, global_side: int) -> dict:
    """Project one prediction from global preprocessed image to original image coordinates."""
    pad_x, pad_y = [float(v) for v in global_meta["pad_xy"]]
    scale = float(global_meta["scale"])
    x0, y0, _, _ = [float(v) for v in crop_meta["bbox_xyxy"]]

    def map_xy(xg: float, yg: float) -> tuple[float, float]:
        xp = (xg / max(scale, 1e-9)) - pad_x
        yp = (yg / max(scale, 1e-9)) - pad_y
        xo = xp + x0
        yo = yp + y0
        return float(np.clip(xo, 0, orig_w - 1)), float(np.clip(yo, 0, orig_h - 1))

    x1g, y1g, x2g, y2g = [float(v) for v in pred["bbox_xyxy"]]
    x1o, y1o = map_xy(x1g, y1g)
    x2o, y2o = map_xy(x2g, y2g)

    poly_o: list[float] = []
    for i in range(0, len(pred["polygon"]), 2):
        xg = float(pred["polygon"][i]) * global_side
        yg = float(pred["polygon"][i + 1]) * global_side
        xo, yo = map_xy(xg, yg)
        poly_o.extend([xo / max(orig_w, 1), yo / max(orig_h, 1)])

    out = dict(pred)
    out["bbox_xyxy"] = [x1o, y1o, x2o, y2o]
    out["polygon"] = poly_o
    return out


@dataclass
class _BranchBundle:
    """Single model branch and its postprocess metadata."""

    name: str
    model: torch.nn.Module
    class_names: list[str]
    image_size: int
    class_thresholds: dict[str, float]
    class_post_cfg: dict[str, dict]
    post_cfg: dict[str, Any]
    device: torch.device

    @classmethod
    def from_infer_config(cls, name: str, infer_cfg_path: str, device: torch.device) -> "_BranchBundle":
        """Instantiate a branch from an existing stage6 inference config."""
        cfg = load_yaml(infer_cfg_path)
        ckpt = torch.load(cfg["model"]["checkpoint"], map_location="cpu")
        class_names = ckpt.get("class_names") or load_yaml(cfg["model"]["class_map_active"]).get("categories", [])
        image_size = int(cfg["model"].get("image_size", 1024))

        vit = load_retfound_vit(cfg["model"]["vendor_dir"], image_size=image_size)
        load_retfound_weights(vit, cfg["model"]["retfound_ckpt"])
        model = RetFoundMaskModel(
            encoder=RetFoundEncoder(vit),
            num_classes=len(class_names),
            decoder_channels=int(cfg["model"].get("decoder_channels", 256)),
        )
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(device).eval()

        post_cfg = dict(cfg.get("postprocess", {}))
        class_thresholds = {n: float(post_cfg.get("threshold", 0.5)) for n in class_names}
        t_json = post_cfg.get("thresholds_json")
        if t_json and Path(t_json).exists():
            t_data = json.loads(Path(t_json).read_text(encoding="utf-8"))
            for n, row in t_data.get("classes", {}).items():
                if n in class_thresholds and "best_threshold" in row:
                    class_thresholds[n] = float(row["best_threshold"])

        class_post_cfg: dict[str, dict] = {}
        p_json = post_cfg.get("class_postprocess_json")
        if p_json and Path(p_json).exists():
            p_data = json.loads(Path(p_json).read_text(encoding="utf-8"))
            class_post_cfg = {str(k): dict(v) for k, v in p_data.get("classes", {}).items() if isinstance(v, dict)}

        return cls(
            name=name,
            model=model,
            class_names=[str(x) for x in class_names],
            image_size=image_size,
            class_thresholds=class_thresholds,
            class_post_cfg=class_post_cfg,
            post_cfg=post_cfg,
            device=device,
        )

    def predict_global(self, image_rgb: np.ndarray) -> list[dict]:
        """Predict detections on one 1024x1024 global preprocessed image."""
        h0, w0 = image_rgb.shape[:2]
        tta_scales = [float(s) for s in self.post_cfg.get("tta_scales", [1.0])]
        tta_hflip = bool(self.post_cfg.get("tta_hflip", False))
        prob_acc = np.zeros((len(self.class_names), h0, w0), dtype=np.float32)
        tta_n = 0
        with torch.no_grad():
            for scale in tta_scales:
                infer_size = max(64, int(round((self.image_size * scale) / 16.0) * 16))
                if hasattr(self.model.encoder.vit.patch_embed, "img_size"):
                    self.model.encoder.vit.patch_embed.img_size = (infer_size, infer_size)
                inp = cv2.resize(image_rgb, (infer_size, infer_size), interpolation=cv2.INTER_AREA)
                ten = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
                prob = torch.sigmoid(self.model(ten))[0].detach().cpu().numpy()
                for c in range(len(self.class_names)):
                    prob_acc[c] += cv2.resize(prob[c], (w0, h0), interpolation=cv2.INTER_LINEAR)
                tta_n += 1
                if tta_hflip:
                    tenf = torch.flip(ten, dims=[3])
                    probf = torch.sigmoid(self.model(tenf))[0].detach().cpu().numpy()[:, :, ::-1]
                    for c in range(len(self.class_names)):
                        prob_acc[c] += cv2.resize(probf[c], (w0, h0), interpolation=cv2.INTER_LINEAR)
                    tta_n += 1

        prob_agg = prob_acc / max(1, tta_n)
        preds: list[dict] = []
        for c, cls_name in enumerate(self.class_names, start=1):
            cls_cfg = dict(self.post_cfg)
            cls_cfg.update(self.class_post_cfg.get(cls_name, {}))
            if "threshold" not in self.class_post_cfg.get(cls_name, {}):
                cls_cfg["threshold"] = float(self.class_thresholds.get(cls_name, cls_cfg.get("threshold", 0.5)))
            preds.extend(_extract_components(prob=prob_agg[c - 1], cls_id=c, cls_name=cls_name, cfg=cls_cfg))

        preds = _nms(preds, iou_thr=float(self.post_cfg.get("nms_iou", 0.3)))
        preds = _cap_per_class(preds, max_per_class=int(self.post_cfg.get("max_preds_per_class", 0)))
        return preds


class InferenceService:
    """Preprocess + multi-branch class-wise inference runtime."""

    def __init__(self, cfg_path: str | Path):
        self.cfg = load_yaml(cfg_path)
        self.device = torch.device(self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        prep = self.cfg["preprocess"]
        self.roi_cfg = dict(prep["roi"])
        self.dataset_name = str(prep.get("dataset_name", "uwf700"))
        self.global_size = int(prep.get("global_size", 1024))
        self.stats_erode_px = int(prep.get("stats_erode_px", 4))
        self.norm_method = str(prep.get("norm_method", "reinhard_lab_ref"))
        self.ref_stats = json.loads(Path(prep["norm_ref_stats_path"]).read_text(encoding="utf-8"))
        self.sam2_masker = Sam2PromptMasker(dict(self.roi_cfg.get("sam2", {})))

        self.branches = {
            name: _BranchBundle.from_infer_config(name, path, self.device)
            for name, path in self.cfg["branches"].items()
        }
        self.profiles = dict(self.cfg["profiles"])

    def list_profiles(self) -> dict:
        """Return profile metadata."""
        return self.profiles

    def _preprocess(self, image_rgb: np.ndarray) -> dict:
        """Run SAM2 ROI + normalization + global resize on one image."""
        mask = compute_roi_mask(
            image=image_rgb,
            cfg=self.roi_cfg,
            dataset=self.dataset_name,
            sam2_masker=self.sam2_masker,
            sam_masker=None,
        )
        crop, crop_meta = crop_to_roi(image_rgb, mask, int(self.roi_cfg.get("crop_pad_px", 12)))
        x0, y0, x1, y1 = crop_meta["bbox_xyxy"]
        roi_crop = mask[y0:y1, x0:x1]
        stats_mask = _safe_erode(roi_crop, self.stats_erode_px)
        norm, norm_meta = normalize_color(
            image=crop,
            stats_mask=stats_mask,
            method=self.norm_method,
            out_mask=roi_crop,
            ref=self.ref_stats,
        )
        global_img, global_meta = resize_global(norm, self.global_size)
        return {
            "mask": mask,
            "crop_meta": crop_meta,
            "norm_meta": norm_meta,
            "global_meta": global_meta,
            "global_img": global_img,
            "roi_crop_mask": roi_crop,
        }

    def predict(self, image_bytes: bytes, profile_name: str = "best_overfit") -> dict:
        """Predict one uploaded image and return JSON-ready artifacts."""
        t0 = perf_counter()
        image_rgb = _decode_image_bytes(image_bytes)
        prep = self._preprocess(image_rgb)
        t1 = perf_counter()

        profile = self.profiles.get(profile_name)
        if profile is None:
            raise ValueError(f"Unknown profile: {profile_name}")
        default_branch = str(profile["default_branch"])
        class_branch = {str(k): str(v) for k, v in profile.get("class_branch", {}).items()}

        needed = {default_branch, *class_branch.values()}
        by_branch: dict[str, list[dict]] = {}
        for name in sorted(needed):
            by_branch[name] = self.branches[name].predict_global(prep["global_img"])
        t2 = perf_counter()

        merged_global: list[dict] = []
        for p in by_branch[default_branch]:
            if class_branch.get(str(p["class_name"]), default_branch) == default_branch:
                merged_global.append(p)
        for cls_name, branch_name in class_branch.items():
            for p in by_branch[branch_name]:
                if str(p["class_name"]) == cls_name:
                    merged_global.append(p)

        oh, ow = image_rgb.shape[:2]
        merged_orig = [
            _convert_global_pred_to_original(
                pred=p,
                crop_meta=prep["crop_meta"],
                global_meta=prep["global_meta"],
                orig_w=ow,
                orig_h=oh,
                global_side=self.global_size,
            )
            for p in merged_global
        ]
        merged_orig = _nms(merged_orig, iou_thr=0.2)
        merged_orig = _cap_per_class(merged_orig, max_per_class=2)
        t3 = perf_counter()

        overlay_orig = _draw_predictions(image_rgb, merged_orig)
        overlay_global = _draw_predictions(prep["global_img"], merged_global)
        roi_vis = np.repeat(prep["mask"][:, :, None], 3, axis=2)

        counts: dict[str, int] = {}
        for p in merged_orig:
            counts[str(p["class_name"])] = counts.get(str(p["class_name"]), 0) + 1

        return {
            "profile": profile_name,
            "predictions": merged_orig,
            "counts_by_class": counts,
            "timings_ms": {
                "preprocess": round((t1 - t0) * 1000.0, 1),
                "model_inference": round((t2 - t1) * 1000.0, 1),
                "merge_postprocess": round((t3 - t2) * 1000.0, 1),
                "total": round((t3 - t0) * 1000.0, 1),
            },
            "images": {
                "original_overlay_png_b64": _encode_png_b64(overlay_orig),
                "global_overlay_png_b64": _encode_png_b64(overlay_global),
                "global_preprocessed_png_b64": _encode_png_b64(prep["global_img"]),
                "roi_mask_png_b64": _encode_png_b64(roi_vis),
            },
        }
