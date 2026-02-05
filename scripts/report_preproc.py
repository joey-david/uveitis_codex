#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse

from uveitis_pipeline.reports import report_preproc


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing report")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preproc-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-n", type=int, default=24)
    args = parser.parse_args()

    out = report_preproc(args.manifest, args.preproc_root, args.out_dir, args.sample_n)
    print(out)


if __name__ == "__main__":
    main()
