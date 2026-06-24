"""Streamlit app for automatic parking space occupancy detection."""
import io
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import MODEL_PATH, ensure_dirs
from detector import get_model, detect, annotate_image

st.set_page_config(page_title="Parking Space Detector", layout="wide", page_icon="P")

ensure_dirs()


@st.cache_resource
def load_model_cached():
    if not MODEL_PATH.exists():
        return None
    return get_model()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Parking Detector")
    st.markdown("---")

    mode = st.radio("Mode", ["Upload Image", "Live Camera"])

    st.markdown("---")

    threshold = st.slider("Confidence Threshold", 0.25, 0.95, 0.50, 0.05)
    show_conf = st.toggle("Show Confidence Labels", value=True)


# ── Model guard ────────────────────────────────────────────────────────────────
model = load_model_cached()
if model is None:
    st.error(
        f"**Model not found** at `{MODEL_PATH}`.\n\n"
        "Train it first (see COLAB_GUIDE.md):\n"
        "```\npython src/convert_to_yolo.py\npython src/train_yolo.py\n```"
    )
    st.stop()


# ── Mode 1: Upload Image ───────────────────────────────────────────────────────
if mode == "Upload Image":
    st.header("Upload Image")

    uploaded = st.file_uploader("Upload a parking lot image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img_pil = Image.open(uploaded).convert("RGB")
        img_np = np.array(img_pil)

        detections = detect(img_np, threshold)

        if not detections:
            st.warning("No parking spaces detected in this image.")
            st.stop()

        annotated = annotate_image(img_np.copy(), detections, show_conf)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(img_pil, use_container_width=True)
        with col2:
            st.subheader("Detected")
            st.image(annotated, use_container_width=True)

        free = sum(1 for d in detections if d["label"] == "Empty")
        occupied = len(detections) - free

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Spaces", len(detections))
        c2.metric("Free", free)
        c3.metric("Occupied", occupied)
        c4.metric("Occupancy", f"{occupied/len(detections)*100:.0f}%" if detections else "N/A")

        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="PNG")
        st.download_button(
            "Download Annotated Image",
            buf.getvalue(),
            file_name="annotated.png",
            mime="image/png"
        )


# ── Mode 2: Live Camera ────────────────────────────────────────────────────────
elif mode == "Live Camera":
    st.header("Live Camera")
    st.info("Live camera requires local installation. Not available on cloud deployments.")

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start Camera") and not st.session_state.camera_running:
            st.session_state.camera_running = True
    with col_stop:
        if st.button("Stop Camera"):
            st.session_state.camera_running = False

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    if st.session_state.camera_running:
        import cv2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera. Make sure a webcam is connected.")
            st.session_state.camera_running = False
        else:
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Frame capture failed.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detect(frame_rgb, threshold)

                if detections:
                    annotated = annotate_image(frame_rgb.copy(), detections, show_conf)
                    frame_placeholder.image(annotated, channels="RGB", use_container_width=True)

                    free = sum(1 for d in detections if d["label"] == "Empty")
                    occ = len(detections) - free
                    stats_placeholder.markdown(
                        f"`Free: {free}` | `Occupied: {occ}` | `Total: {len(detections)}`"
                    )
                else:
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    stats_placeholder.markdown("No spaces detected in this frame.")

                time.sleep(0.2)  # ~5 FPS

            cap.release()
