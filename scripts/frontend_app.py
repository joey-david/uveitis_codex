#!/usr/bin/env python3
"""Streamlit frontend for pipeline visualization and inference."""

from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from pathlib import Path
import sys

import requests
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uveitis_pipeline.common import load_yaml
from uveitis_pipeline.frontend import FrontendInferenceService


def _decode_png_b64(encoded: str) -> Image.Image:
    """Decode a base64 PNG payload into a PIL image."""
    return Image.open(BytesIO(base64.b64decode(encoded)))


def _inject_css() -> None:
    """Inject a bold, responsive visual style."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg0: #f2f7f9;
  --bg1: #dceff4;
  --ink: #16313a;
  --muted: #3f5f68;
  --accent: #008f73;
  --accent-soft: #6dd3c3;
  --warm: #ff7a3d;
  --card: #ffffffee;
}

html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 700px at 5% -20%, var(--accent-soft), transparent),
              linear-gradient(125deg, var(--bg0), var(--bg1));
  color: var(--ink);
}

h1, h2, h3, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2 {
  font-family: "Space Grotesk", sans-serif;
  letter-spacing: 0.01em;
}

p, li, label, [data-testid="stCaptionContainer"], [data-testid="stSidebar"] {
  font-family: "IBM Plex Mono", monospace;
}

.block-container {
  max-width: 1180px;
  padding-top: 1.2rem;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f5fbfd, #e7f4f8);
  border-right: 1px solid #b8d8e2;
}

.front-card {
  background: var(--card);
  border: 1px solid #b8d8e2;
  border-radius: 14px;
  padding: 14px 16px;
  margin: 8px 0 16px;
  box-shadow: 0 14px 24px rgba(28, 66, 77, 0.08);
}

.front-step {
  color: var(--muted);
  font-size: 0.9rem;
  margin-bottom: 0.3rem;
}

.front-title {
  color: var(--ink);
  font-size: 1.05rem;
  font-weight: 700;
}

.front-badge {
  display: inline-block;
  margin: 0.2rem 0.4rem 0.6rem 0;
  padding: 0.2rem 0.5rem;
  border-radius: 999px;
  font-size: 0.72rem;
  border: 1px solid #9ac5d2;
  background: #ecf8fb;
  color: #1f4e5a;
}

.front-loader {
  height: 8px;
  border-radius: 999px;
  overflow: hidden;
  background: #d9ecef;
  position: relative;
}

.front-loader::after {
  content: "";
  position: absolute;
  top: 0;
  left: -35%;
  height: 100%;
  width: 35%;
  background: linear-gradient(90deg, var(--accent), var(--warm));
  animation: frontslide 1.1s infinite;
}

@keyframes frontslide {
  0% { left: -35%; }
  100% { left: 100%; }
}

@media (max-width: 768px) {
  .block-container { padding-left: 0.7rem; padding-right: 0.7rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _get_local_service(config_path: str) -> FrontendInferenceService:
    """Cache a local inference service for repeated runs."""
    return FrontendInferenceService.from_yaml(config_path)


def _run_remote(
    endpoint: str,
    file_name: str,
    file_bytes: bytes,
    score_thresh: float,
    dataset: str,
    timeout_sec: int,
) -> dict:
    """Call remote API and return parsed JSON response."""
    url = endpoint.rstrip("/") + "/infer"
    resp = requests.post(
        url,
        files={"file": (file_name, file_bytes, "application/octet-stream")},
        data={"score_thresh": str(score_thresh), "dataset": dataset},
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    return resp.json()


def _step_card(step: str, title: str, body: str = "") -> None:
    """Render one step card header."""
    st.markdown(
        f"""
<div class="front-card">
  <div class="front-step">{step}</div>
  <div class="front-title">{title}</div>
  <div>{body}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_prediction_table(preds: list[dict]) -> None:
    """Render compact prediction rows."""
    if not preds:
        st.info("No detections above threshold.")
        return
    rows = []
    for p in preds:
        x1, y1, x2, y2 = p["box_xyxy"]
        rows.append(
            {
                "label": p["label_name"],
                "score": round(float(p["score"]), 4),
                "x1": int(round(x1)),
                "y1": int(round(y1)),
                "x2": int(round(x2)),
                "y2": int(round(y2)),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    """Render and run the Streamlit app."""
    st.set_page_config(page_title="Uveitis Pipeline Frontend", page_icon="\U0001fa7a", layout="wide")
    _inject_css()

    config_path = os.environ.get("UVEITIS_FRONTEND_CONFIG", "configs/frontend.yaml")
    cfg = load_yaml(config_path)
    frontend_cfg = cfg.get("frontend", {})
    runtime_cfg = cfg.get("runtime", {})
    pipeline_cfg = cfg.get("pipeline", {})

    st.title("Uveitis Pipeline Visual Frontend")
    st.caption("Upload one image, inspect SAM ROI selection, then review final detector labels on the original view.")

    default_mode = str(frontend_cfg.get("mode", "local")).lower()
    mode_options = ["local", "remote"]
    mode_idx = mode_options.index(default_mode) if default_mode in mode_options else 0

    with st.sidebar:
        st.markdown("### Runtime")
        mode = st.radio("Inference Backend", options=mode_options, index=mode_idx, horizontal=True)
        remote_url = st.text_input("Remote API URL", value=str(frontend_cfg.get("remote_api_url", "http://localhost:8000")))
        dataset_name = st.text_input("ROI Dataset Token", value=str(pipeline_cfg.get("dataset_for_roi", "uwf700")))
        score_thresh = st.slider(
            "Score Threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(runtime_cfg.get("score_thresh", 0.3)),
            step=0.01,
        )
        timeout_sec = int(frontend_cfg.get("remote_timeout_sec", 180))
        st.markdown(
            f"<span class='front-badge'>config: {config_path}</span>"
            f"<span class='front-badge'>timeout: {timeout_sec}s</span>",
            unsafe_allow_html=True,
        )

    upload = st.file_uploader("Select fundus image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    run = st.button("Run Pipeline", type="primary", use_container_width=True, disabled=upload is None)

    if upload is None:
        _step_card("Step 1", "Image Selection", "Choose an image to start the 4-step pipeline view.")
        return

    _step_card("Step 1", "Image Selection", f"Selected: `{upload.name}`")
    st.image(upload.getvalue(), use_container_width=True)

    if not run:
        return

    if mode == "local":
        service = _get_local_service(config_path)

    with st.spinner("Processing... SAM ROI + normalization + detector"):
        if mode == "local":
            result = service.infer_bytes(upload.getvalue(), image_name=upload.name, score_thresh=score_thresh, dataset=dataset_name)
        else:
            result = _run_remote(
                endpoint=remote_url,
                file_name=upload.name,
                file_bytes=upload.getvalue(),
                score_thresh=score_thresh,
                dataset=dataset_name,
                timeout_sec=timeout_sec,
            )

    _step_card("Step 2", "SAM ROI Inference", "The mask and overlay show what region is analyzed.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(_decode_png_b64(result["images"]["roi_overlay"]), caption="ROI Boundary on Input", use_container_width=True)
    with c2:
        st.image(_decode_png_b64(result["images"]["roi_mask"]), caption="Binary ROI Mask", use_container_width=True)
    with c3:
        st.image(_decode_png_b64(result["images"]["masked_raw"]), caption="Fundus-only (masked raw)", use_container_width=True)

    _step_card("Step 3", "Waiting / Processing Animation", "Detector stage progression.")
    st.markdown("<div class='front-loader'></div>", unsafe_allow_html=True)
    p = st.progress(0, text="Scoring labels on ROI...")
    for i in range(1, 101, 20):
        time.sleep(0.07)
        p.progress(i, text="Scoring labels on ROI...")
    p.empty()

    _step_card("Step 4", "Final Labels on Original Image", "Initial image is juxtaposed with final predictions.")
    o1, o2 = st.columns(2)
    with o1:
        st.image(_decode_png_b64(result["images"]["input"]), caption="Initial Image", use_container_width=True)
    with o2:
        st.image(_decode_png_b64(result["images"]["final_overlay"]), caption="Final Labels", use_container_width=True)

    st.image(_decode_png_b64(result["images"]["juxtaposed"]), caption="Juxtaposed Comparison", use_container_width=True)

    st.markdown("#### Predictions")
    _render_prediction_table(result.get("predictions", []))

    timings = result.get("timings_sec", {})
    st.caption(
        f"run_id={result.get('run_id', '')} | total={timings.get('total', 0.0):.2f}s | "
        f"artifact_dir={result.get('artifact_dir', '(disabled)')}"
    )


if __name__ == "__main__":
    main()
