#!/usr/bin/env python3
"""Run one Main9 OBB train/eval experiment and emit a compact JSON summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def _read_best_epoch(results_csv: Path) -> tuple[int | None, float | None, float | None]:
    """Return best epoch and mAPs from a Ultralytics results.csv file."""
    if not results_csv.exists():
        return None, None, None
    rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
    if not rows:
        return None, None, None
    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    return int(float(best["epoch"])), float(best["metrics/mAP50(B)"]), float(best["metrics/mAP50-95(B)"])


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Train and evaluate one Main9 OBB experiment.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True, help="Initial checkpoint or model name.")
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", default="out/yolo_obb/uwf700_tiles_main9/data.yaml")
    ap.add_argument("--project", default="runs/sweeps_main9")
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--lr0", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--cos-lr", action="store_true")
    ap.add_argument("--warmup-epochs", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--close-mosaic", type=int, default=None)
    ap.add_argument("--freeze", type=int, default=None)
    ap.add_argument("--mosaic", type=float, default=None)
    ap.add_argument("--mixup", type=float, default=None)
    ap.add_argument("--copy-paste", type=float, default=None)
    ap.add_argument("--degrees", type=float, default=None)
    ap.add_argument("--translate", type=float, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--hsv-h", type=float, default=None)
    ap.add_argument("--hsv-s", type=float, default=None)
    ap.add_argument("--hsv-v", type=float, default=None)
    ap.add_argument("--fliplr", type=float, default=None)
    ap.add_argument("--flipud", type=float, default=None)
    ap.add_argument("--erasing", type=float, default=None)
    ap.add_argument("--eval-imgsz", type=int, default=1536)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    run_dir = Path(args.project) / args.name
    kwargs: dict[str, object] = {
        "data": args.train_data,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "project": args.project,
        "name": args.name,
    }
    if args.cos_lr:
        kwargs["cos_lr"] = True
    if args.weight_decay is not None:
        kwargs["weight_decay"] = args.weight_decay
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.deterministic:
        kwargs["deterministic"] = True
    if args.warmup_epochs is not None:
        kwargs["warmup_epochs"] = args.warmup_epochs
    if args.patience is not None:
        kwargs["patience"] = args.patience
    if args.close_mosaic is not None:
        kwargs["close_mosaic"] = args.close_mosaic
    if args.freeze is not None:
        kwargs["freeze"] = args.freeze
    if args.mosaic is not None:
        kwargs["mosaic"] = args.mosaic
    if args.mixup is not None:
        kwargs["mixup"] = args.mixup
    if args.copy_paste is not None:
        kwargs["copy_paste"] = args.copy_paste
    if args.degrees is not None:
        kwargs["degrees"] = args.degrees
    if args.translate is not None:
        kwargs["translate"] = args.translate
    if args.scale is not None:
        kwargs["scale"] = args.scale
    if args.hsv_h is not None:
        kwargs["hsv_h"] = args.hsv_h
    if args.hsv_s is not None:
        kwargs["hsv_s"] = args.hsv_s
    if args.hsv_v is not None:
        kwargs["hsv_v"] = args.hsv_v
    if args.fliplr is not None:
        kwargs["fliplr"] = args.fliplr
    if args.flipud is not None:
        kwargs["flipud"] = args.flipud
    if args.erasing is not None:
        kwargs["erasing"] = args.erasing

    model = YOLO(args.model)
    train_out = model.train(**kwargs)
    save_dir = None
    if hasattr(train_out, "save_dir"):
        save_dir = Path(str(train_out.save_dir))
    elif getattr(model, "trainer", None) is not None and hasattr(model.trainer, "save_dir"):
        save_dir = Path(str(model.trainer.save_dir))
    if save_dir is not None:
        run_dir = save_dir
    best_path = run_dir / "weights" / "best.pt"
    val = YOLO(best_path.as_posix()).val(
        data=args.val_data,
        split="val",
        imgsz=args.eval_imgsz,
        batch=args.eval_batch,
        device=args.device,
        workers=args.workers,
        verbose=False,
        project="/tmp",
        name=f"val_{args.name}",
    )

    best_epoch, train_best_map50, train_best_map5095 = _read_best_epoch(run_dir / "results.csv")
    summary = {
        "name": args.name,
        "run_dir": run_dir.as_posix(),
        "best_weights": best_path.as_posix(),
        "train_data": args.train_data,
        "val_data": args.val_data,
        "params": {
            "model": args.model,
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "batch": args.batch,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "deterministic": bool(args.deterministic),
            "cos_lr": bool(args.cos_lr),
            "warmup_epochs": args.warmup_epochs,
            "patience": args.patience,
            "close_mosaic": args.close_mosaic,
            "freeze": args.freeze,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "copy_paste": args.copy_paste,
            "degrees": args.degrees,
            "translate": args.translate,
            "scale": args.scale,
            "hsv_h": args.hsv_h,
            "hsv_s": args.hsv_s,
            "hsv_v": args.hsv_v,
            "fliplr": args.fliplr,
            "flipud": args.flipud,
            "erasing": args.erasing,
        },
        "train_best_epoch": best_epoch,
        "train_best_map50": train_best_map50,
        "train_best_map50_95": train_best_map5095,
        "eval_map50": float(val.box.map50),
        "eval_map50_95": float(val.box.map),
        "eval_class_map50_95": [float(x) for x in getattr(val.box, "maps", [])],
    }
    out_json = args.out_json or Path("eval/sweeps") / f"{args.name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
