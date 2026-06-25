"""Full detection pipeline: image -> annotated image + stats, via YOLOv8."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import cv2
from PIL import Image

from utils import MODEL_PATH, COLOR_FREE_BGR, COLOR_OCCUPIED_BGR

_model = None


def get_model(model_path: Path = MODEL_PATH):
    """Lazily load and cache the YOLO model."""
    global _model
    if _model is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Train it first: python src/convert_to_yolo.py && python src/train_yolo.py"
            )
        from ultralytics import YOLO
        _model = YOLO(str(model_path))
    return _model


def detect(image_rgb: np.ndarray, threshold: float = 0.5, model_path: Path = MODEL_PATH) -> list:
    """Run detection on an RGB image. Returns list of dicts with box, label, confidence."""
    model = get_model(model_path)
    # Ultralytics treats raw numpy array inputs as BGR (OpenCV convention) and
    # flips them to RGB internally, so feed it BGR to undo that correctly.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = model.predict(image_bgr, conf=threshold, verbose=False)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = "Occupied" if model.names[class_id] == "occupied" else "Empty"
        detections.append({
            "x": int(x1), "y": int(y1),
            "w": int(x2 - x1), "h": int(y2 - y1),
            "label": label,
            "confidence": confidence,
        })
    return detections


def annotate_image(image: np.ndarray, detections: list, show_confidence: bool = True) -> np.ndarray:
    """Draw colored bounding boxes with optional confidence labels on image."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        draw = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        draw = image.copy()

    overlay = draw.copy()

    for idx, det in enumerate(detections, start=1):
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        color = COLOR_FREE_BGR if det["label"] == "Empty" else COLOR_OCCUPIED_BGR

        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.3, draw, 0.7, 0, draw)
        overlay = draw.copy()

        cv2.rectangle(draw, (x, y), (x + w, y + h), color, 2)

        if show_confidence:
            text = f"#{idx} {det['confidence']:.0%}"
            text_y = max(y - 5, 15)
            cv2.putText(draw, text, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)


def run_detection(image_path: Path, threshold: float = 0.5, show_confidence: bool = True) -> tuple:
    """Full pipeline: image file -> annotated image (RGB) + stats dict."""
    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    detections = detect(img_rgb, threshold)
    annotated = annotate_image(img_rgb, detections, show_confidence)

    free = sum(1 for d in detections if d["label"] == "Empty")
    occupied = len(detections) - free
    stats = {
        "total": len(detections),
        "free": free,
        "occupied": occupied,
        "occupancy_pct": (occupied / len(detections) * 100) if detections else 0.0
    }

    return annotated, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run parking space detection on an image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="annotated.jpg")
    args = parser.parse_args()

    annotated, stats = run_detection(Path(args.image), args.threshold)
    Image.fromarray(annotated).save(args.output)
    print(f"Results: {stats}")
    print(f"Saved to {args.output}")
