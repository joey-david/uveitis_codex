#!/usr/bin/env python3
"""Run main detector hypothesis sweeps from YAML config."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def _cmd_from_exp(exp: dict, cfg: dict, out_json: Path) -> list[str]:
    """Build run_main9_experiment command for one experiment."""
    cmd = [
        "python",
        "scripts/run_main9_experiment.py",
        "--name",
        str(exp["name"]),
        "--model",
        str(exp["model"]),
        "--train-data",
        str(exp["train_data"]),
        "--val-data",
        str(cfg["subagent_c"]["val_data"]),
        "--project",
        str(cfg["subagent_c"]["project"]),
        "--imgsz",
        str(exp.get("imgsz", 1536)),
        "--epochs",
        str(exp.get("epochs", 4)),
        "--batch",
        str(exp.get("batch", 4)),
        "--device",
        str(cfg["general"].get("device", "0")),
        "--workers",
        str(cfg["general"].get("workers", 8)),
        "--optimizer",
        str(exp.get("optimizer", "AdamW")),
        "--lr0",
        str(exp.get("lr0", 2e-4)),
        "--eval-imgsz",
        str(cfg["subagent_c"].get("eval_imgsz", 1536)),
        "--eval-batch",
        str(cfg["subagent_c"].get("eval_batch", 4)),
        "--out-json",
        out_json.as_posix(),
    ]
    if exp.get("seed") is not None:
        cmd += ["--seed", str(exp["seed"])]
    if exp.get("cos_lr", False):
        cmd.append("--cos-lr")
    for key in [
        "weight_decay",
        "warmup_epochs",
        "patience",
        "close_mosaic",
        "freeze",
        "mosaic",
        "mixup",
        "copy_paste",
        "degrees",
        "translate",
        "scale",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "fliplr",
        "flipud",
        "erasing",
    ]:
        if exp.get(key) is not None:
            cmd += [f"--{key.replace('_', '-')}", str(exp[key])]
    return cmd


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Run hypothesis sweep for main detector.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, default=Path("out/logs/diagnostics"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    exps = cfg["subagent_c"]["experiments"]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for exp in exps:
        out_json = Path(cfg["general"]["out_root"]) / "hypothesis_sweeps" / f"{exp['name']}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"{exp['name']}.log"
        cmd = _cmd_from_exp(exp, cfg, out_json)
        with log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        rec = {
            "name": exp["name"],
            "returncode": int(proc.returncode),
            "out_json": out_json.as_posix(),
            "log": log_path.as_posix(),
        }
        if out_json.exists():
            rec["metrics"] = json.loads(out_json.read_text(encoding="utf-8"))
        results.append(rec)

    args.out_json.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
