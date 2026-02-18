#!/usr/bin/env python3
"""Extract plain text from a PDF (best-effort) for quick searching/notes."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from pypdf import PdfReader

    r = PdfReader(str(args.pdf))
    parts: list[str] = []
    for i, p in enumerate(r.pages):
        parts.append(f"\n\n===== page {i + 1} =====\n{p.extract_text() or ''}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

