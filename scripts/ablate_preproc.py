#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse

from uveitis_pipeline.reports import ablate_preproc


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare predictions with/without preprocessing")
    parser.add_argument("--pred-a", required=True)
    parser.add_argument("--pred-b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = ablate_preproc(args.pred_a, args.pred_b, args.out)
    print(out)


if __name__ == "__main__":
    main()
