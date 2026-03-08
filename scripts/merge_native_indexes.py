#!/usr/bin/env python3
"""Merge native index jsonl files with optional repeats."""

from __future__ import annotations

import argparse
from pathlib import Path
import json


def main() -> None:
    """Merge multiple jsonl indexes into one output file."""
    p = argparse.ArgumentParser(description="Merge jsonl indexes")
    p.add_argument("--input", action="append", required=True, help="Path or path:repeat (e.g. a.jsonl:3)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = []
    for spec in args.input:
        if ":" in spec:
            path_s, rep_s = spec.rsplit(":", 1)
            rep = max(1, int(rep_s))
        else:
            path_s, rep = spec, 1
        path = Path(path_s)
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for _ in range(rep):
            rows.extend(lines)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out": out.as_posix(), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
