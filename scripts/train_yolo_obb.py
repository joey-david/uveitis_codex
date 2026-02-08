#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8m-obb.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/yolo_obb")
    ap.add_argument("--name", required=True)
    ap.add_argument("--optimizer", default=None, help="Override Ultralytics optimizer (e.g. SGD, AdamW).")
    ap.add_argument("--lr0", type=float, default=None, help="Override initial LR (ignored if optimizer=auto).")
    ap.add_argument("--cos-lr", action="store_true", help="Use cosine LR schedule.")
    ap.add_argument("--freeze", type=int, default=None, help="Freeze first N layers.")
    args = ap.parse_args()

    kwargs = {}
    if args.optimizer:
        kwargs["optimizer"] = args.optimizer
    if args.lr0 is not None:
        kwargs["lr0"] = args.lr0
    if args.cos_lr:
        kwargs["cos_lr"] = True
    if args.freeze is not None:
        kwargs["freeze"] = args.freeze

    YOLO(args.model).train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        **kwargs,
    )


if __name__ == "__main__":
    main()
