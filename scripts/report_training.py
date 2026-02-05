#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse

from uveitis_pipeline.reports import report_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Training curves and summary report")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-png", required=True)
    args = parser.parse_args()

    out = report_training(args.run_dir, args.out_json, args.out_png)
    print(out)


if __name__ == "__main__":
    main()
