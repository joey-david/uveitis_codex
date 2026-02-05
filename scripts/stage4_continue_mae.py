#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional: continue RetFound MAE pretraining on UWF")
    parser.add_argument("--retfound-dir", default="../RETFound")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="pretrain_runs/mae_uwf")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    retfound_dir = Path(args.retfound_dir).resolve()
    if not retfound_dir.exists():
        repo = Path(__file__).resolve().parents[1]
        for alt in [
            (repo / "RETFound").resolve(),
            (repo / "retfound").resolve(),
            (repo / "third_party" / "RETFound").resolve(),
            (repo / "third_party" / "retfound").resolve(),
            (repo / "third_party" / "RETFound_MAE").resolve(),
        ]:
            if alt.exists():
                retfound_dir = alt
                break
    pretrain_script = retfound_dir / "main_pretrain.py"

    cmd = [
        "python",
        str(pretrain_script),
        "--data_path",
        args.data_path,
        "--output_dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
    ]

    if not pretrain_script.exists():
        print(
            {
                "status": "skipped",
                "reason": f"{pretrain_script} not found in RETFound repo",
                "next_step": "If you add MAE pretrain code, rerun this script with --run",
                "command_template": " ".join(cmd),
            }
        )
        return

    print({"status": "ready", "command": " ".join(cmd)})
    if args.run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
