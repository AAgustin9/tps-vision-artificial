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
from detector import get_car_model, detect, annotate_image, make_grid_slots, classify_slots

st.set_page_config(page_title="Parking Space Detector", layout="wide", page_icon="P")

ensure_dirs()


@st.cache_resource
def load_vehicle_model_cached():
    return get_car_model()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Parking Detector")
    st.markdown("---")

    mode = st.radio("Mode", ["Upload Image", "Live Camera"])

    st.markdown("---")

    threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.35, 0.05)
    show_conf = st.toggle("Show Confidence Labels", value=True)
    detection_strategy = st.radio(
        "Detection Strategy",
        ["Automatic", "Manual Grid"],
        help="Use Manual Grid when automatic line detection duplicates or hallucinates spaces."
    )


# ── Model guard ────────────────────────────────────────────────────────────────
try:
    load_vehicle_model_cached()
except Exception as exc:
    st.error(f"Could not load the pretrained vehicle detector: {exc}")
    st.stop()

if not MODEL_PATH.exists():
    st.sidebar.warning(
        f"Optional trained parking model not found at `{MODEL_PATH}`. "
        "The app will use line-based slot detection + pretrained vehicle detection."
    )


# ── Mode 1: Upload Image ───────────────────────────────────────────────────────
if mode == "Upload Image":
    st.header("Upload Image")

    uploaded = st.file_uploader("Upload a parking lot image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img_pil = Image.open(uploaded).convert("RGB")
        img_np = np.array(img_pil)

        if detection_strategy == "Manual Grid":
            h, w = img_np.shape[:2]
            st.info(
                "Manual Grid is the reliable mode for fixed cameras or difficult images: "
                "adjust the rectangle so it covers the parking row/area, set rows/columns, "
                "then occupancy is classified inside those calibrated spaces."
            )
            with st.expander("Manual grid calibration", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    grid_cols = st.number_input("Columns / spaces per row", 1, 50, 5, 1)
                    grid_rows = st.number_input("Rows", 1, 10, 1, 1)
                with c2:
                    grid_x = st.slider("Grid X", 0, max(0, w - 1), int(w * 0.08))
                    grid_y = st.slider("Grid Y", 0, max(0, h - 1), int(h * 0.05))
                with c3:
                    max_grid_w = max(1, w - grid_x)
                    max_grid_h = max(1, h - grid_y)
                    grid_w = st.slider("Grid Width", 1, max_grid_w, min(max_grid_w, max(1, int(w * 0.85))))
                    grid_h = st.slider("Grid Height", 1, max_grid_h, min(max_grid_h, max(1, int(h * 0.65))))

            slots = make_grid_slots(img_np.shape, grid_x, grid_y, grid_w, grid_h, grid_cols, grid_rows)
            detections = classify_slots(img_np, slots)
        else:
            detections = detect(img_np, threshold)

        if not detections:
            st.warning("No parking spaces detected in this image. Try Manual Grid mode for this image/camera.")
            st.stop()

        used_manual = any(d.get("source") == "manual+occupancy" for d in detections)
        used_hybrid = any(d.get("source") in {"slot+car", "slot+visual"} for d in detections)
        used_fallback = any(d.get("source") == "line-fallback" for d in detections)
        if used_manual:
            st.success("Using calibrated manual grid + occupancy detection.")
        elif used_hybrid:
            st.info("Using painted-line slot detection + pretrained vehicle detection.")
        elif used_fallback:
            st.info("Using painted-line fallback detection for visible empty stalls.")

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
                    fallback_note = " | `slot+car/visual`" if any(d.get("source") in {"slot+car", "slot+visual"} for d in detections) else " | `line fallback`" if any(d.get("source") == "line-fallback" for d in detections) else ""
                    stats_placeholder.markdown(
                        f"`Free: {free}` | `Occupied: {occ}` | `Total: {len(detections)}`{fallback_note}"
                    )
                else:
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    stats_placeholder.markdown("No spaces detected in this frame.")

                time.sleep(0.2)  # ~5 FPS

            cap.release()
