#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uveitis_pipeline.vascularite import (
    VesselMaskParams,
    bbox_to_poly_px,
    extract_patch,
    nonblack_mask,
    obb_norm_to_poly_px,
    point_in_any_poly,
    vessel_mask,
)


class PatchDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int, int, int]], patch: int, augment: bool = False, seed: int = 0):
        self.samples = samples
        self.patch = int(patch)
        self.augment = bool(augment)
        self.rng = random.Random(int(seed))
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _read(self, path: str) -> np.ndarray:
        if path not in self._cache:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            self._cache[path] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._cache[path]

    def __getitem__(self, idx: int):
        path, x, y, lab = self.samples[idx]
        rgb = self._read(path)
        patch = extract_patch(rgb, int(x), int(y), self.patch)
        if self.augment:
            # Lightweight color/contrast jitter to help generalization.
            a = self.rng.uniform(0.85, 1.20)
            b = self.rng.uniform(-12.0, 12.0)
            patch = np.clip(patch.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        x_t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        y_t = torch.tensor([float(lab)], dtype=torch.float32)
        return x_t, y_t


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
        return timm.create_model(timm_name, pretrained=True, num_classes=1, in_chans=3)
    raise ValueError(f"Unknown model: {name}")


def _load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_samples(
    coco: dict,
    out_cache_dir: Path,
    patch: int,
    pos_per_img: int,
    neg_per_img: int,
    seed: int,
    vessel_params: VesselMaskParams,
) -> list[tuple[str, int, int, int]]:
    random.seed(int(seed))
    out_cache_dir.mkdir(parents=True, exist_ok=True)

    cats = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    vascular_ids = {cid for cid, name in cats.items() if name == "vascularite"}
    if not vascular_ids:
        raise ValueError("No category named 'vascularite' in COCO categories")

    by_img = defaultdict(list)
    for a in coco.get("annotations", []):
        by_img[int(a["image_id"])].append(a)

    samples: list[tuple[str, int, int, int]] = []
    for im in coco.get("images", []):
        path = str(im["file_name"])
        if not Path(path).exists():
            continue
        rgb_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        nb = nonblack_mask(rgb)
        vm = vessel_mask(rgb, nb=nb, p=vessel_params)
        ys, xs = np.where(vm > 0)
        if xs.size < 64:
            continue

        # Heuristic: vascular "sheathing" is bright around (dark) vessels.
        # Build a thin bright ring around vessel pixels to reduce noisy positives from OBB supervision.
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0].astype(np.float32)
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        ring = (cv2.dilate(vm, k2, iterations=1) > 0) & ~(cv2.dilate(vm, k1, iterations=1) > 0)
        ring = ring & (nb > 0)
        vals = L[ring]
        if vals.size >= 64:
            t_bright = float(np.quantile(vals, 0.90))
            sheath = (ring & (L >= t_bright)).astype(np.uint8)
            pos_hint = (cv2.dilate(sheath, k1, iterations=1) > 0) & (vm > 0)
        else:
            pos_hint = vm > 0

        polys: list[np.ndarray] = []
        for a in by_img.get(int(im["id"]), []):
            if int(a["category_id"]) not in vascular_ids:
                continue
            obb = a.get("obb")
            if isinstance(obb, list) and len(obb) == 8:
                polys.append(obb_norm_to_poly_px([float(v) for v in obb], w=w, h=h))
            else:
                polys.append(bbox_to_poly_px(a["bbox"]))

        gt = np.zeros((h, w), dtype=np.uint8)
        for poly in polys:
            cv2.fillPoly(gt, [poly.astype(np.int32)], 1)

        pos_mask = (pos_hint & (gt > 0)).astype(np.uint8)
        neg_mask = ((vm > 0) & (gt == 0)).astype(np.uint8)

        pos = []
        neg = []
        ys_p, xs_p = np.where(pos_mask > 0)
        ys_n, xs_n = np.where(neg_mask > 0)
        if polys:
            if xs_p.size >= 8:
                idx = list(range(xs_p.size))
                random.shuffle(idx)
                for j in idx[: int(pos_per_img)]:
                    pos.append((path, int(xs_p[j]), int(ys_p[j]), 1))
            else:
                continue
        if xs_n.size >= 8:
            idx = list(range(xs_n.size))
            random.shuffle(idx)
            for j in idx[: int(neg_per_img)]:
                neg.append((path, int(xs_n[j]), int(ys_n[j]), 0))

        if polys and len(pos) < max(4, int(pos_per_img) // 8):
            continue
        samples.extend(pos)
        samples.extend(neg)

    random.shuffle(samples)
    (out_cache_dir / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True, help="COCO json with obb for vascularite (typically global_1024).")
    ap.add_argument("--out", type=Path, default=Path("runs/vascularite_patch"))
    ap.add_argument("--name", type=str, default="patch64")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--pos-per-img", type=int, default=64)
    ap.add_argument("--neg-per-img", type=int, default=128)
    ap.add_argument("--model", type=str, default="timm:resnet18")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = args.out / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    coco = _load_coco(args.coco)
    vessel_params = VesselMaskParams()

    samples = _build_samples(
        coco=coco,
        out_cache_dir=run_dir,
        patch=int(args.patch),
        pos_per_img=int(args.pos_per_img),
        neg_per_img=int(args.neg_per_img),
        seed=int(args.seed),
        vessel_params=vessel_params,
    )
    if len(samples) < 256:
        raise RuntimeError(f"Too few samples: {len(samples)}")

    by_path = defaultdict(list)
    for s in samples:
        by_path[s[0]].append(s)
    paths = sorted(by_path.keys())
    random.Random(int(args.seed)).shuffle(paths)
    n_tr = max(1, int(0.8 * len(paths)))
    tr_paths = set(paths[:n_tr])
    tr = [s for s in samples if s[0] in tr_paths]
    va = [s for s in samples if s[0] not in tr_paths]

    ds_tr = PatchDataset(tr, patch=int(args.patch), augment=bool(args.augment), seed=int(args.seed))
    ds_va = PatchDataset(va, patch=int(args.patch), augment=False, seed=int(args.seed) + 1)
    dl_tr = DataLoader(ds_tr, batch_size=int(args.batch), shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=int(args.batch), shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device(str(args.device))
    model = build_model(str(args.model), patch=int(args.patch)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best = 1e9
    log_path = run_dir / "metrics.jsonl"
    t_start = time.time()

    for epoch in range(int(args.epochs)):
        model.train()
        loss_tr = 0.0
        n_tr = 0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            loss_tr += float(loss.item()) * xb.size(0)
            n_tr += xb.size(0)

        model.eval()
        loss_va = 0.0
        n_va = 0
        acc = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                p = (torch.sigmoid(logits) >= 0.5).float()
                acc += float((p == yb).float().sum().item())
                loss_va += float(loss.item()) * xb.size(0)
                n_va += xb.size(0)
        loss_tr /= max(1, n_tr)
        loss_va /= max(1, n_va)
        acc /= max(1, n_va)

        rec = {"epoch": epoch, "train_loss": loss_tr, "val_loss": loss_va, "val_acc": acc}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec)

        if loss_va < best:
            best = loss_va
            torch.save(
                {"model": model.state_dict(), "model_name": str(args.model), "patch": int(args.patch)},
                run_dir / "best.pt",
            )

    dt = time.time() - t_start
    (run_dir / "done.txt").write_text(f"seconds={dt:.1f}\n", encoding="utf-8")
    print(f"Saved: {run_dir/'best.pt'}")


if __name__ == "__main__":
    main()
