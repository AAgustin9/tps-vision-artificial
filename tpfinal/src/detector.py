"""Full detection pipeline: image + lot config → annotated image + stats."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import numpy as np
import cv2
from PIL import Image
from utils import LOT_CONFIG_PATH, COLOR_FREE_BGR, COLOR_OCCUPIED_BGR, DEFAULT_CONFIGS
from classifier import get_classifier


def load_lot_config(lot_name: str, config_path: Path = LOT_CONFIG_PATH) -> list:
    """Load space definitions for a named lot from config JSON."""
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIGS

    if lot_name not in config:
        available = list(config.keys())
        raise KeyError(f"Lot '{lot_name}' not found. Available: {available}")

    return config[lot_name]["spaces"]


def crop_space(image: np.ndarray, space: dict) -> np.ndarray:
    """Crop a single parking space ROI from the image."""
    h, w = image.shape[:2]
    x = max(0, int(space["x"]))
    y = max(0, int(space["y"]))
    x2 = min(w, x + int(space["w"]))
    y2 = min(h, y + int(space["h"]))
    return image[y:y2, x:x2]


def annotate_image(
    image: np.ndarray,
    spaces: list,
    results: list,
    show_confidence: bool = True
) -> np.ndarray:
    """Draw colored bounding boxes with optional confidence labels on image."""
    annotated = image.copy()

    # Convert RGB to BGR for OpenCV drawing
    if len(annotated.shape) == 3 and annotated.shape[2] == 3:
        draw = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    else:
        draw = annotated

    overlay = draw.copy()

    for space, (label, confidence) in zip(spaces, results):
        x, y, w, h = int(space["x"]), int(space["y"]), int(space["w"]), int(space["h"])
        color = COLOR_FREE_BGR if label == "Empty" else COLOR_OCCUPIED_BGR

        # Semi-transparent fill
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.3, draw, 0.7, 0, draw)
        overlay = draw.copy()

        # Solid border
        cv2.rectangle(draw, (x, y), (x + w, y + h), color, 2)

        if show_confidence:
            text = f"#{space['id']} {confidence:.0%}"
            text_y = max(y - 5, 15)
            cv2.putText(draw, text, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Convert back to RGB
    return cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)


def run_detection(
    image_path: Path,
    lot_name: str,
    threshold: float = 0.7,
    show_confidence: bool = True
) -> tuple:
    """Full pipeline: image file → annotated image (RGB) + stats dict."""
    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    spaces = load_lot_config(lot_name)
    crops = [crop_space(img_rgb, s) for s in spaces]

    classifier = get_classifier(threshold)
    results = classifier.predict_batch(crops)

    annotated = annotate_image(img_rgb, spaces, results, show_confidence)

    free = sum(1 for label, _ in results if label == "Empty")
    occupied = len(results) - free
    stats = {
        "total": len(results),
        "free": free,
        "occupied": occupied,
        "occupancy_pct": (occupied / len(results) * 100) if results else 0.0
    }

    return annotated, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run parking space detection on an image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--lot-name", required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output", default="annotated.jpg")
    args = parser.parse_args()

    annotated, stats = run_detection(Path(args.image), args.lot_name, args.threshold)
    Image.fromarray(annotated).save(args.output)
    print(f"Results: {stats}")
    print(f"Saved to {args.output}")
