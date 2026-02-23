#!/usr/bin/env python3
"""Dispatch diagnostics workstreams with CPU/GPU-aware scheduling."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml


def _start(cmd: list[str], log_path: Path) -> subprocess.Popen:
    """Start one subprocess and redirect stdout/stderr to log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)


def _run(cmd: list[str], log_path: Path) -> int:
    """Run one subprocess synchronously and store logs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode)


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Dispatch diagnostics subagents.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-sweeps", action="store_true")
    ap.add_argument("--status-json", type=Path, default=Path("eval/diagnostics/dispatch_status.json"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_root = Path(cfg["general"]["out_root"])
    logs = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    dataset_cmd = [
        "python",
        "scripts/diagnostics/subagent_a_data_qc.py",
        "--out-json",
        (out_root / "data_qc" / "datasets_integrity.json").as_posix(),
    ]
    for p in cfg["subagent_a"]["datasets"]:
        dataset_cmd += ["--data-yaml", str(p)]

    preproc_cmd = [
        "python",
        "scripts/diagnostics/subagent_b_preproc_qc.py",
        "--roi-dir",
        str(cfg["subagent_b"]["roi_dir"]),
        "--norm-dir",
        str(cfg["subagent_b"]["norm_dir"]),
        "--sample-limit",
        str(cfg["subagent_b"].get("sample_limit", 0)),
        "--out-json",
        (out_root / "preproc_ablation" / "mask_norm_qc.json").as_posix(),
    ]

    main_eval_cmd = [
        "python",
        "scripts/diagnostics/subagent_d_model_eval.py",
        "--model",
        str(cfg["baseline"]["main"]["model"]),
        "--data",
        str(cfg["baseline"]["main"]["data"]),
        "--imgsz",
        str(cfg["baseline"]["imgsz"]),
        "--batch",
        str(cfg["baseline"]["batch"]),
        "--conf",
        str(cfg["baseline"]["conf"]),
        "--iou",
        str(cfg["baseline"]["iou"]),
        "--max-det",
        str(cfg["baseline"]["max_det"]),
        "--duplicate-iou",
        str(cfg["baseline"]["duplicate_iou"]),
        "--device",
        str(cfg["general"]["device"]),
        "--out-json",
        (out_root / "baseline_repro" / "main_baseline.json").as_posix(),
    ]

    vasc_eval_cmd = [
        "python",
        "scripts/diagnostics/subagent_d_model_eval.py",
        "--model",
        str(cfg["baseline"]["vascularite"]["model"]),
        "--data",
        str(cfg["baseline"]["vascularite"]["data"]),
        "--imgsz",
        str(cfg["baseline"]["imgsz"]),
        "--batch",
        str(cfg["baseline"]["batch"]),
        "--conf",
        str(cfg["baseline"]["conf"]),
        "--iou",
        str(cfg["baseline"]["iou"]),
        "--max-det",
        str(cfg["baseline"]["max_det"]),
        "--duplicate-iou",
        str(cfg["baseline"]["duplicate_iou"]),
        "--device",
        str(cfg["general"]["device"]),
        "--out-json",
        (out_root / "baseline_repro" / "vascularite_baseline.json").as_posix(),
    ]

    cpu_procs = {
        "subagent_a_data_qc": _start(dataset_cmd, logs / "subagent_a_data_qc.log"),
        "subagent_b_preproc_qc": _start(preproc_cmd, logs / "subagent_b_preproc_qc.log"),
    }

    status = {}
    status["subagent_d_main_eval"] = _run(main_eval_cmd, logs / "subagent_d_main_eval.log")
    status["subagent_d_vascularite_eval"] = _run(vasc_eval_cmd, logs / "subagent_d_vascularite_eval.log")

    if args.run_sweeps:
        sweep_cmd = [
            "python",
            "scripts/diagnostics/subagent_c_sweeps.py",
            "--config",
            args.config.as_posix(),
            "--out-json",
            (out_root / "hypothesis_sweeps" / "sweep_summary.json").as_posix(),
            "--log-dir",
            (logs / "subagent_c").as_posix(),
        ]
        status["subagent_c_sweeps"] = _run(sweep_cmd, logs / "subagent_c_dispatch.log")

    for name, proc in cpu_procs.items():
        status[name] = int(proc.wait())

    report = {
        "config": args.config.as_posix(),
        "run_sweeps": bool(args.run_sweeps),
        "status": status,
        "duration_sec": float(time.time() - started_at),
        "logs_dir": logs.as_posix(),
        "out_root": out_root.as_posix(),
    }
    args.status_json.parent.mkdir(parents=True, exist_ok=True)
    args.status_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
