#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse

from uveitis_pipeline.common import load_yaml
from uveitis_pipeline.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN detector from YAML config")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    report = train_from_config(cfg)
    print(report)


if __name__ == "__main__":
    main()
