#!/usr/bin/env python3
"""Serve single-image frontend inference over HTTP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uveitis_pipeline.frontend import FrontendInferenceService


def build_app(config_path: str) -> FastAPI:
    """Build a FastAPI app bound to one frontend config."""
    service = FrontendInferenceService.from_yaml(config_path)
    app = FastAPI(title="Uveitis Frontend Inference API", version="1.0")

    @app.get("/health")
    async def health() -> dict:
        """Return a compact readiness payload."""
        return {
            "status": "ok",
            "device": str(service.device),
            "checkpoint": str(service.infer_cfg["model"]["checkpoint"]),
            "default_score_thresh": float(service.default_score_thresh),
            "detector_input_size": int(service.detector_size),
        }

    @app.post("/infer")
    async def infer(
        file: UploadFile = File(...),
        score_thresh: float | None = Form(None),
        dataset: str | None = Form(None),
    ) -> dict:
        """Run full pipeline inference and return serialized outputs."""
        raw = await file.read()
        return service.infer_bytes(raw, image_name=file.filename or "upload.png", score_thresh=score_thresh, dataset=dataset)

    return app


def main() -> None:
    """Run the HTTP server."""
    parser = argparse.ArgumentParser(description="Run frontend inference API")
    parser.add_argument("--config", default="configs/frontend.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = build_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
