"""Streamlit app for parking space occupancy detection."""
import sys
import io
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import MODEL_PATH, LOT_CONFIG_PATH, ensure_dirs
from space_selector import load_config, save_config
from detector import load_lot_config, crop_space, annotate_image

st.set_page_config(page_title="Parking Space Detector", layout="wide", page_icon="P")

ensure_dirs()


@st.cache_resource
def load_model_cached(threshold: float):
    if not MODEL_PATH.exists():
        return None
    from classifier import ParkingClassifier
    return ParkingClassifier(threshold=threshold)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Parking Detector")
    st.markdown("---")

    mode = st.radio("Mode", ["Upload Image", "Live Camera"])

    lot_config = load_config()
    lot_names = list(lot_config.keys())
    selected_lot = st.selectbox("Lot Configuration", lot_names)

    st.markdown("---")

    threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.70, 0.05)
    show_conf = st.toggle("Show Confidence Labels", value=True)

    st.markdown("---")
    st.subheader("Lot Config")

    new_lot_name = st.text_input("New Lot Name")
    define_btn = st.button("Define New Lot (CLI)")

    uploaded_config = st.file_uploader("Load Config JSON", type="json")
    if uploaded_config is not None:
        new_cfg = json.load(uploaded_config)
        lot_config.update(new_cfg)
        save_config(lot_config)
        st.success("Config loaded and saved.")

    if st.button("Save Current Config"):
        save_config(lot_config)
        st.success("Config saved.")


# ── Model guard ────────────────────────────────────────────────────────────────
classifier = load_model_cached(threshold)
if classifier is None:
    st.error(
        f"**Model not found** at `{MODEL_PATH}`.\n\n"
        "Please train the model first:\n"
        "```\npython src/prepare_dataset.py\npython src/train.py\n```"
    )
    if define_btn and new_lot_name:
        st.info(
            f"To define lot **{new_lot_name}**, run from terminal:\n\n"
            f"```\npython src/space_selector.py --image <path> --lot-name {new_lot_name}\n```"
        )
    st.stop()


# ── Define new lot info ────────────────────────────────────────────────────────
if define_btn and new_lot_name:
    st.sidebar.warning(
        f"Run from terminal to define '{new_lot_name}':\n\n"
        f"`python src/space_selector.py --image <img> --lot-name {new_lot_name}`"
    )


# ── Mode 1: Upload Image ───────────────────────────────────────────────────────
if mode == "Upload Image":
    st.header("Upload Image")

    uploaded = st.file_uploader("Upload a parking lot image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img_pil = Image.open(uploaded).convert("RGB")
        img_np = np.array(img_pil)

        try:
            spaces = load_lot_config(selected_lot)
        except KeyError as e:
            st.error(str(e))
            st.stop()

        crops = [crop_space(img_np, s) for s in spaces]

        valid_pairs = [(s, c) for s, c in zip(spaces, crops) if c.size > 0]
        if not valid_pairs:
            st.warning("No valid space crops found. Check the lot configuration dimensions.")
            st.stop()

        v_spaces, v_crops = zip(*valid_pairs)
        results = classifier.predict_batch(list(v_crops))
        annotated = annotate_image(img_np.copy(), list(v_spaces), results, show_conf)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(img_pil, use_container_width=True)
        with col2:
            st.subheader("Detected")
            st.image(annotated, use_container_width=True)

        free = sum(1 for label, _ in results if label == "Empty")
        occupied = len(results) - free

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Spaces", len(results))
        c2.metric("Free", free)
        c3.metric("Occupied", occupied)
        c4.metric("Occupancy", f"{occupied/len(results)*100:.0f}%" if results else "N/A")

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

        try:
            spaces = load_lot_config(selected_lot)
        except KeyError as e:
            st.error(str(e))
            st.session_state.camera_running = False
            st.stop()

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
                crops = [crop_space(frame_rgb, s) for s in spaces]
                valid_pairs = [(s, c) for s, c in zip(spaces, crops) if c.size > 0]

                if valid_pairs:
                    v_spaces, v_crops = zip(*valid_pairs)
                    results = classifier.predict_batch(list(v_crops))
                    annotated = annotate_image(frame_rgb.copy(), list(v_spaces), results, show_conf)
                    frame_placeholder.image(annotated, channels="RGB", use_container_width=True)

                    free = sum(1 for l, _ in results if l == "Empty")
                    occ = len(results) - free
                    stats_placeholder.markdown(
                        f"`Free: {free}` | `Occupied: {occ}` | `Total: {len(results)}`"
                    )
                else:
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                time.sleep(0.2)  # ~5 FPS

            cap.release()
