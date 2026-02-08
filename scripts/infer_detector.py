#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse

from uveitis_pipeline.common import load_yaml
from uveitis_pipeline.infer import run_inference_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tile inference + global merge")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_inference_from_config(cfg)


if __name__ == "__main__":
    main()
