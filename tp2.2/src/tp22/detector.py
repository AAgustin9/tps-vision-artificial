from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .descriptors import (
    extract_external_contours,
    is_supported_image,
    iter_images,
    load_image,
    preprocess_image,
    contour_to_hu_moments,
)


@dataclass(slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bounding_box: BoundingBox
    contour_area: float
    perimeter: float
    hu_moments: list[float]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"] = round(self.confidence, 4)
        data["contour_area"] = round(self.contour_area, 2)
        data["perimeter"] = round(self.perimeter, 2)
        data["hu_moments"] = [round(value, 6) for value in self.hu_moments]
        return data


@dataclass(slots=True)
class ProcessedImage:
    image_name: str
    detections: list[Detection]
    annotated_image: np.ndarray
    binary_mask: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image_name,
            "detections": [detection.to_dict() for detection in self.detections],
        }


def classify_contour(contour: np.ndarray, model_bundle: dict[str, Any]) -> tuple[str, float, np.ndarray]:
    model = model_bundle["model"]
    features = contour_to_hu_moments(contour)
    prediction = model.predict([features])[0]

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([features])[0]
        confidence = float(np.max(probabilities))

    return str(prediction), confidence, features


def detect_shapes(
    image: np.ndarray,
    model_bundle: dict[str, Any],
    min_area: int = 50,
) -> tuple[list[Detection], np.ndarray]:
    binary = preprocess_image(image)
    contours = extract_external_contours(binary, min_area=min_area)
    detections: list[Detection] = []

    for contour in contours:
        label, confidence, features = classify_contour(contour, model_bundle)
        x, y, width, height = cv2.boundingRect(contour)
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
                contour_area=cv2.contourArea(contour),
                perimeter=cv2.arcLength(contour, True),
                hu_moments=features.tolist(),
            )
        )

    detections.sort(key=lambda item: (item.bounding_box.y, item.bounding_box.x, item.label))
    return detections, binary


def process_image(
    image_path: Path,
    model_bundle: dict[str, Any],
    min_area: int = 50,
) -> ProcessedImage:
    from .drawing import draw_detections

    image = load_image(image_path)
    detections, binary = detect_shapes(image, model_bundle, min_area=min_area)
    annotated = draw_detections(image, detections)
    return ProcessedImage(
        image_name=image_path.name,
        detections=detections,
        annotated_image=annotated,
        binary_mask=binary,
    )
