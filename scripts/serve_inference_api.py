#!/usr/bin/env python3
"""Serve REST API for UWF inference."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import os

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from uveitis_pipeline.inference_service import InferenceService


def _build_app(service: InferenceService) -> FastAPI:
    """Build FastAPI app around the shared inference service."""
    app = FastAPI(title="Uveitis Inference API", version="1.0.0")

    origins_env = os.getenv("UVEITIS_CORS_ORIGINS", "*")
    origins = [x.strip() for x in origins_env.split(",") if x.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    token = os.getenv("UVEITIS_API_TOKEN", "").strip()

    def _auth(authorization: str | None) -> None:
        if not token:
            return
        want = f"Bearer {token}"
        if authorization != want:
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/health")
    def health() -> dict:
        """Healthcheck endpoint."""
        return {"ok": True}

    @app.get("/v1/profiles")
    def list_profiles(authorization: str | None = Header(default=None)) -> dict:
        """List available inference profiles."""
        _auth(authorization)
        return {"profiles": service.list_profiles()}

    @app.post("/v1/predict")
    async def predict(
        file: UploadFile = File(...),
        profile: str = Form(default="best_overfit"),
        authorization: str | None = Header(default=None),
    ) -> dict:
        """Run one-image inference and return detections + overlays."""
        _auth(authorization)
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Upload must be an image")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        try:
            return service.predict(image_bytes=data, profile_name=profile)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    """Entrypoint for API server."""
    parser = argparse.ArgumentParser(description="Serve UWF inference API")
    parser.add_argument("--config", default="configs/inference_api.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    service = InferenceService(args.config)
    app = _build_app(service)
    uvicorn.run(app, host=args.host, port=int(args.port), workers=1)


if __name__ == "__main__":
    main()
