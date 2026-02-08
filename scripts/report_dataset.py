#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse

from uveitis_pipeline.reports import report_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset + label bottleneck report")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--cocos", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = report_dataset(args.manifests, args.cocos, args.out)
    print(out)


if __name__ == "__main__":
    main()
