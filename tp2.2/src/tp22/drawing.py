from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection


COLORS = {
    "circle": (0, 200, 255),
    "rectangle_outline": (0, 220, 0),
    "star": (255, 160, 0),
}


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    canvas = image.copy()
    for detection in detections:
        box = detection.bounding_box
        color = COLORS.get(detection.label, (255, 255, 255))
        cv2.rectangle(
            canvas,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            color,
            2,
        )
        caption = f"{detection.label} ({detection.confidence:.2f})"
        origin = (box.x, max(18, box.y - 8))
        cv2.putText(
            canvas,
            caption,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas
