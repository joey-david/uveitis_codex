#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uveitis_pipeline.common import ensure_dir, read_image, write_image
from uveitis_pipeline.vascularite import (
    VesselMaskParams,
    extract_patch,
    nonblack_mask,
    obb_norm_to_poly_px,
    vessel_mask,
)


def build_model(name: str, patch: int) -> nn.Module:
    name = str(name).lower()
    if name == "tiny":
        net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        feat = 64 * (int(patch) // 8) * (int(patch) // 8)
        head = nn.Sequential(nn.Flatten(), nn.Linear(feat, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

        class _Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = net
                self.head = head

            def forward(self, x):
                return self.head(self.net(x))

        return _Tiny()
    if name.startswith("timm:") or name in ("resnet18", "resnet34", "efficientnet_b0"):
        import timm

        timm_name = name.split("timm:", 1)[-1]
        return timm.create_model(timm_name, pretrained=False, num_classes=1, in_chans=3)
    raise ValueError(f"Unknown model: {name}")


def _load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _draw_poly(img: np.ndarray, poly: np.ndarray, color=(255, 0, 0), thickness: int = 2) -> np.ndarray:
    out = img.copy()
    pts = poly.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=int(thickness))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--coco", type=Path, required=True, help="COCO global_1024 json (for image list + GT overlays)")
    ap.add_argument("--out", type=Path, default=Path("eval/vascularite_patch_previews"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--stride", type=int, default=10, help="Sampling stride on vessel pixels (higher = faster, less dense).")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--model", type=str, default="", help="Override model name stored in checkpoint (e.g. timm:resnet18)")
    ap.add_argument("--thresh", type=float, default=0.60)
    ap.add_argument("--min-area", type=int, default=80)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    coco = _load_coco(args.coco)
    out_dir = ensure_dir(args.out)
    out_masks = ensure_dir(out_dir / "masks")
    out_vis = ensure_dir(out_dir / "overlays")
    out_preds = ensure_dir(out_dir / "preds")

    cats = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    vascular_ids = {cid for cid, name in cats.items() if name == "vascularite"}
    by_img = defaultdict(list)
    for a in coco.get("annotations", []):
        by_img[int(a["image_id"])].append(a)

    device = torch.device(str(args.device))
    state = torch.load(args.weights, map_location="cpu")
    model_name = str(args.model or state.get("model_name") or "tiny")
    patch = int(args.patch or state.get("patch") or 64)
    model = build_model(model_name, patch=patch).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    vessel_params = VesselMaskParams()

    ims = coco.get("images", [])[: int(args.n)]
    for im in ims:
        path = Path(im["file_name"])
        if not path.exists():
            continue

        rgb = read_image(path)
        h, w = rgb.shape[:2]
        nb = nonblack_mask(rgb)
        vm = vessel_mask(rgb, nb=nb, p=vessel_params)

        ys, xs = np.where(vm > 0)
        if xs.size == 0:
            continue
        keep = (xs % int(args.stride) == 0) & (ys % int(args.stride) == 0)
        xs = xs[keep]
        ys = ys[keep]
        if xs.size == 0:
            continue

        batch = 256
        scores = np.zeros((h, w), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, xs.size, batch):
                xb = []
                for x, y in zip(xs[i : i + batch], ys[i : i + batch]):
                    p = extract_patch(rgb, int(x), int(y), int(args.patch))
                    xb.append(torch.from_numpy(p).permute(2, 0, 1).float() / 255.0)
                x_t = torch.stack(xb, dim=0).to(device)
                s = torch.sigmoid(model(x_t)).squeeze(1).detach().cpu().numpy()
                for (x, y, sc) in zip(xs[i : i + batch], ys[i : i + batch], s):
                    scores[int(y), int(x)] = float(sc)

        m = (scores >= float(args.thresh)).astype(np.uint8)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.dilate(m, ker, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=1)
        m = (m & (nb > 0)).astype(np.uint8)

        preds = []
        n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i_cc in range(1, n_cc):
            area = int(stats[i_cc][4])
            if area < int(args.min_area):
                continue
            comp = cc == i_cc
            ys2, xs2 = np.where(comp)
            if xs2.size < 3:
                continue
            pts = np.stack([xs2.astype(np.float32), ys2.astype(np.float32)], axis=1)
            rect = cv2.minAreaRect(pts)
            poly = cv2.boxPoints(rect).astype(np.float32)
            score = float(scores[comp].max()) if comp.any() else 0.0
            obb = [float(v) for xy in zip(poly[:, 0] / w, poly[:, 1] / h) for v in xy]
            preds.append({"obb": obb, "score": score, "area": area})
        preds.sort(key=lambda d: d["score"], reverse=True)
        preds = preds[:50]

        key = str(im.get("image_id") or path.stem).replace("::", "__")
        write_image(out_masks / f"{key}.png", (m * 255).astype(np.uint8))
        (out_preds / f"{key}.json").write_text(json.dumps({"image_id": im.get("image_id"), "predictions": preds}), encoding="utf-8")

        vis = rgb.copy()
        for p in preds[:15]:
            poly = obb_norm_to_poly_px(p["obb"], w=w, h=h)
            vis = _draw_poly(vis, poly, color=(255, 0, 0), thickness=2)
        for a in by_img.get(int(im["id"]), []):
            if int(a["category_id"]) not in vascular_ids:
                continue
            obb = a.get("obb")
            if isinstance(obb, list) and len(obb) == 8:
                poly = obb_norm_to_poly_px([float(v) for v in obb], w=w, h=h)
                vis = _draw_poly(vis, poly, color=(0, 255, 0), thickness=2)
        write_image(out_vis / f"{key}.png", vis)

    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
