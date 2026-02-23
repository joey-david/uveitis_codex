"""Frontend inference helpers for single-image pipeline visualization."""

from __future__ import annotations

import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from torchvision.transforms import functional as TF

from .common import draw_boxes, ensure_dir, load_yaml, save_json, write_image
from .infer import _load_model
from .preprocess import Sam2PromptMasker, SamPromptMasker, _safe_erode, compute_roi_mask, crop_to_roi, normalize_color


def _encode_png_b64(image: np.ndarray) -> str:
    """Encode an RGB image to base64 PNG."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode PNG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _overlay_boundary(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return image with ROI boundary highlighted."""
    out = image.copy()
    edges = cv2.Canny(mask.astype(np.uint8), 80, 160)
    out[edges > 0] = np.array([255, 90, 0], dtype=np.uint8)
    return out


def _as_rgb_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a binary mask to RGB."""
    m = (mask > 0).astype(np.uint8) * 255
    return np.repeat(m[:, :, None], 3, axis=2)


def _masked_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Black out non-ROI pixels."""
    out = image.copy()
    out[mask <= 0] = 0
    return out


def _read_image_bytes(raw: bytes) -> np.ndarray:
    """Decode uploaded bytes into an RGB image."""
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(pil, dtype=np.uint8)


def _safe_slug(name: str) -> str:
    """Build a compact path-safe slug."""
    keep = [c if c.isalnum() or c in ("-", "_") else "_" for c in Path(name).stem.lower()]
    return "".join(keep).strip("_")[:48] or "image"


def _juxtapose(left: np.ndarray, right: np.ndarray, left_title: str, right_title: str) -> np.ndarray:
    """Create a side-by-side comparison panel."""
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    canvas = np.zeros((h + 42, w, 3), dtype=np.uint8)
    canvas[:, :, :] = np.array([242, 246, 247], dtype=np.uint8)
    canvas[42 : 42 + left.shape[0], : left.shape[1]] = left
    canvas[42 : 42 + right.shape[0], left.shape[1] :] = right
    cv2.line(canvas, (left.shape[1], 42), (left.shape[1], h + 41), (180, 190, 195), 2)
    cv2.putText(canvas, left_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (35, 44, 50), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        right_title,
        (left.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (35, 44, 50),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _classwise_nms(pred: dict, iou_thresh: float) -> dict:
    """Apply class-wise NMS to one detector output."""
    boxes = pred["boxes"].cpu()
    scores = pred["scores"].cpu()
    labels = pred["labels"].cpu()
    keep_all = []
    for cls in torch.unique(labels).tolist():
        idx = torch.where(labels == int(cls))[0]
        keep = nms(boxes[idx], scores[idx], float(iou_thresh))
        keep_all.extend(idx[keep].tolist())
    if not keep_all:
        return {"boxes": boxes[:0], "scores": scores[:0], "labels": labels[:0]}
    order = torch.argsort(scores[keep_all], descending=True)
    keep_tensor = torch.tensor(keep_all, dtype=torch.long)[order]
    return {
        "boxes": boxes[keep_tensor],
        "scores": scores[keep_tensor],
        "labels": labels[keep_tensor],
    }


class FrontendInferenceService:
    """Run single-image SAM + detector inference for UI/API."""

    def __init__(self, cfg: dict):
        """Create service from a frontend YAML dictionary."""
        self.cfg = cfg
        self.pipeline_cfg = cfg.get("pipeline", {})
        self.ui_cfg = cfg.get("frontend", {})
        self.runtime_cfg = cfg.get("runtime", {})

        infer_cfg_path = self.pipeline_cfg.get("infer_config", "configs/infer_uveitis_ft.yaml")
        preproc_cfg_path = self.pipeline_cfg.get("preprocess_config", "configs/stage0_preprocess.yaml")

        self.infer_cfg = load_yaml(infer_cfg_path)
        self.preproc_cfg = load_yaml(preproc_cfg_path)
        self.dataset_for_roi = str(self.pipeline_cfg.get("dataset_for_roi", "uwf700"))

        req_device = str(self.runtime_cfg.get("device", "auto")).lower()
        if req_device == "auto":
            req_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(req_device)

        self.model = _load_model(self.infer_cfg["model"]["checkpoint"], self.infer_cfg, self.device)
        self.class_names = self.infer_cfg["model"].get("class_names", [])

        self.roi_cfg = self.preproc_cfg.get("roi", {})
        self.norm_cfg = self.preproc_cfg.get("normalize", {})
        self.ref_stats = self._load_ref_stats(self.norm_cfg)
        self.sam_masker, self.sam2_masker = self._build_maskers(self.roi_cfg)

        self.default_score_thresh = float(self.runtime_cfg.get("score_thresh", self.infer_cfg["runtime"].get("score_thresh", 0.3)))
        self.default_nms_iou = float(self.runtime_cfg.get("nms_iou", self.infer_cfg["runtime"].get("nms_iou", 0.5)))
        model_input = int(self.infer_cfg["model"].get("input_size", 0))
        self.detector_size = int(self.runtime_cfg.get("detector_input_size", model_input if model_input > 0 else 1024))

        out_root = self.ui_cfg.get("artifact_root", "out/frontend")
        self.artifact_root = ensure_dir(Path(out_root))
        self.save_artifacts = bool(self.ui_cfg.get("save_artifacts", True))

    @staticmethod
    def _load_ref_stats(norm_cfg: dict) -> dict | None:
        """Load color reference stats for reinhard normalization."""
        if norm_cfg.get("method") != "reinhard_lab_ref":
            return None
        ref_path = norm_cfg.get("ref", {}).get("stats_path", "")
        if not ref_path:
            return None
        p = Path(ref_path)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def _build_maskers(roi_cfg: dict) -> tuple[SamPromptMasker | None, Sam2PromptMasker | None]:
        """Instantiate SAM/SAM2 maskers with threshold fallback."""
        use_sam = roi_cfg.get("method") == "sam_prompted" or any(
            v == "sam_prompted" for v in roi_cfg.get("method_by_dataset", {}).values()
        )
        use_sam2 = roi_cfg.get("method") == "sam2_prompted" or any(
            v == "sam2_prompted" for v in roi_cfg.get("method_by_dataset", {}).values()
        )

        sam_masker = None
        if use_sam:
            sam_cfg = roi_cfg.get("sam", {})
            try:
                sam_masker = SamPromptMasker(sam_cfg)
            except Exception:
                if not bool(sam_cfg.get("fallback_to_threshold", True)):
                    raise

        sam2_masker = None
        if use_sam2:
            sam2_cfg = roi_cfg.get("sam2", {})
            try:
                sam2_masker = Sam2PromptMasker(sam2_cfg)
            except Exception:
                if not bool(sam2_cfg.get("fallback_to_threshold", True)):
                    raise

        return sam_masker, sam2_masker

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "FrontendInferenceService":
        """Load frontend service from YAML path."""
        return cls(load_yaml(config_path))

    def infer_bytes(
        self,
        raw: bytes,
        image_name: str = "upload.png",
        score_thresh: float | None = None,
        dataset: str | None = None,
    ) -> dict:
        """Run full inference from image bytes."""
        image = _read_image_bytes(raw)
        return self.infer_image(image=image, image_name=image_name, score_thresh=score_thresh, dataset=dataset)

    def infer_image(
        self,
        image: np.ndarray,
        image_name: str = "upload.png",
        score_thresh: float | None = None,
        dataset: str | None = None,
    ) -> dict:
        """Run full inference from an RGB numpy image."""
        t0 = time.time()
        data_tag = _safe_slug(image_name)
        run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}_{data_tag}"
        run_dir = self.artifact_root / run_id
        if self.save_artifacts:
            ensure_dir(run_dir)

        ds_name = dataset or self.dataset_for_roi
        mask = compute_roi_mask(
            image,
            self.roi_cfg,
            dataset=ds_name,
            sam_masker=self.sam_masker,
            sam2_masker=self.sam2_masker,
        )
        roi_overlay = _overlay_boundary(image, mask)
        masked_raw = _masked_image(image, mask)

        crop, crop_meta = crop_to_roi(image, mask, int(self.roi_cfg.get("crop_pad_px", 12)))
        x0, y0, x1, y1 = crop_meta["bbox_xyxy"]
        roi_crop = mask[y0:y1, x0:x1]
        stats_mask = _safe_erode(roi_crop, int(self.norm_cfg.get("stats_erode_px", 4)))
        norm_method = self.norm_cfg.get("method", "clahe_luminance")
        norm, norm_meta = normalize_color(crop, stats_mask, norm_method, out_mask=roi_crop, ref=self.ref_stats)

        detector_input = cv2.resize(norm, (self.detector_size, self.detector_size), interpolation=cv2.INTER_LINEAR)
        t1 = time.time()

        with torch.no_grad():
            tensor = TF.to_tensor(detector_input).to(self.device)
            pred = self.model([tensor])[0]
        pred = _classwise_nms(pred, self.default_nms_iou)

        score_thr = self.default_score_thresh if score_thresh is None else float(score_thresh)
        keep = pred["scores"] >= score_thr
        boxes = pred["boxes"][keep].cpu().numpy().tolist()
        scores = pred["scores"][keep].cpu().numpy().tolist()
        labels = pred["labels"][keep].cpu().numpy().tolist()

        crop_w = max(1, int(crop.shape[1]))
        crop_h = max(1, int(crop.shape[0]))
        sx = crop_w / float(self.detector_size)
        sy = crop_h / float(self.detector_size)

        mapped = []
        vis_labels = []
        for box, score, label in zip(boxes, scores, labels):
            bx = [
                float(box[0] * sx + x0),
                float(box[1] * sy + y0),
                float(box[2] * sx + x0),
                float(box[3] * sy + y0),
            ]
            cls = int(label)
            name = self.class_names[cls - 1] if 0 < cls <= len(self.class_names) else str(cls)
            mapped.append(
                {
                    "label_id": cls,
                    "label_name": name,
                    "score": float(score),
                    "box_xyxy": bx,
                }
            )
            vis_labels.append(f"{name}:{float(score):.2f}")

        final_overlay = draw_boxes(image, [m["box_xyxy"] for m in mapped], vis_labels, color=(0, 184, 116))
        juxtaposed = _juxtapose(image, final_overlay, "Input", "Final Labels")
        t2 = time.time()

        outputs = {
            "input": image,
            "roi_mask": _as_rgb_mask(mask),
            "roi_overlay": roi_overlay,
            "masked_raw": masked_raw,
            "normalized_crop": norm,
            "detector_input": detector_input,
            "final_overlay": final_overlay,
            "juxtaposed": juxtaposed,
        }
        if self.save_artifacts:
            for key, img in outputs.items():
                write_image(run_dir / f"{key}.png", img)
            save_json(
                run_dir / "result.json",
                {
                    "run_id": run_id,
                    "dataset": ds_name,
                    "image_name": image_name,
                    "mask_area_ratio": float((mask > 0).mean()),
                    "crop_meta": crop_meta,
                    "norm_meta": norm_meta,
                    "predictions": mapped,
                    "timings_sec": {
                        "stage1_mask_norm": float(t1 - t0),
                        "stage2_detect": float(t2 - t1),
                        "total": float(t2 - t0),
                    },
                },
            )

        return {
            "run_id": run_id,
            "artifact_dir": run_dir.as_posix() if self.save_artifacts else "",
            "dataset": ds_name,
            "image_name": image_name,
            "mask_area_ratio": float((mask > 0).mean()),
            "crop_meta": crop_meta,
            "norm_meta": norm_meta,
            "runtime": {
                "device": str(self.device),
                "score_thresh": float(score_thr),
                "nms_iou": float(self.default_nms_iou),
                "detector_input_size": int(self.detector_size),
            },
            "timings_sec": {
                "stage1_mask_norm": float(t1 - t0),
                "stage2_detect": float(t2 - t1),
                "total": float(t2 - t0),
            },
            "predictions": mapped,
            "images": {k: _encode_png_b64(v) for k, v in outputs.items()},
        }
