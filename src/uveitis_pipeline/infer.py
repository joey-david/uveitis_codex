from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.ops import nms
from torchvision.transforms import functional as TF

from .common import draw_boxes, ensure_dir, read_image, save_json, write_image
from .modeling import build_detector


def _load_model(ckpt_path: str | Path, cfg: dict, device: torch.device):
    model = build_detector(cfg).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def predict_tiles(
    model,
    image_id: str,
    preproc_root: Path,
    device: torch.device,
    score_thresh: float,
):
    key = image_id.replace("::", "__")
    meta_path = preproc_root / "tiles_meta" / f"{key}.json"
    if not meta_path.exists():
        return []

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    out = []
    with torch.no_grad():
        for tile in meta["tiles"]:
            tile_file = preproc_root / "tiles" / key / f"{tile['tile_id']}.png"
            if not tile_file.exists():
                continue
            image = TF.to_tensor(read_image(tile_file)).to(device)
            pred = model([image])[0]
            keep = pred["scores"] >= score_thresh
            boxes = pred["boxes"][keep].cpu()
            scores = pred["scores"][keep].cpu()
            labels = pred["labels"][keep].cpu()
            for b, s, l in zip(boxes, scores, labels):
                out.append(
                    {
                        "box": [
                            float(b[0] + tile["x0"]),
                            float(b[1] + tile["y0"]),
                            float(b[2] + tile["x0"]),
                            float(b[3] + tile["y0"]),
                        ],
                        "score": float(s),
                        "label": int(l),
                    }
                )
    return out


def merge_tile_preds(tile_preds: list[dict], iou_thresh: float) -> list[dict]:
    merged: list[dict] = []
    by_label: dict[int, list[dict]] = {}
    for p in tile_preds:
        by_label.setdefault(p["label"], []).append(p)

    for label, preds in by_label.items():
        boxes = torch.tensor([p["box"] for p in preds], dtype=torch.float32)
        scores = torch.tensor([p["score"] for p in preds], dtype=torch.float32)
        keep = nms(boxes, scores, iou_thresh)
        for idx in keep.tolist():
            merged.append(preds[idx])

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged


def run_inference_from_config(cfg: dict) -> None:
    preproc_root = Path(cfg["input"]["preproc_root"])
    out_pred = ensure_dir(Path(cfg["output"]["pred_dir"]) / cfg["output"]["exp_name"])
    out_vis = ensure_dir(Path(cfg["output"]["vis_dir"]) / cfg["output"]["exp_name"])

    image_ids_file = cfg["input"].get("image_ids_json")
    if image_ids_file:
        loaded = json.loads(Path(image_ids_file).read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            split_name = cfg["input"].get("split_name") or cfg["runtime"].get("split_name", "test")
            image_ids = loaded.get(split_name, [])
        else:
            image_ids = loaded
    else:
        image_ids = [p.stem.replace("__", "::") for p in sorted((preproc_root / "tiles_meta").glob("*.json"))]

    device = torch.device(cfg["runtime"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(cfg["model"]["checkpoint"], cfg, device)
    class_names = cfg["model"].get("class_names", [])

    for image_id in image_ids:
        preds = predict_tiles(
            model,
            image_id,
            preproc_root,
            device,
            score_thresh=float(cfg["runtime"].get("score_thresh", 0.3)),
        )
        merged = merge_tile_preds(preds, iou_thresh=float(cfg["runtime"].get("nms_iou", 0.5)))

        key = image_id.replace("::", "__")
        save_json(out_pred / f"{key}.json", {"image_id": image_id, "predictions": merged})

        global_img = preproc_root / "global_1024" / f"{key}.png"
        if global_img.exists() and merged:
            img = read_image(global_img)
            boxes = [p["box"] for p in merged]
            labels = []
            for p in merged:
                cls = p["label"]
                name = class_names[cls - 1] if 0 < cls <= len(class_names) else str(cls)
                labels.append(f"{name}:{p['score']:.2f}")
            vis = draw_boxes(img, boxes, labels)
            write_image(out_vis / f"{key}.png", vis)
