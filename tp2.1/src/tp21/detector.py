from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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
    distance: float
    bounding_box: BoundingBox
    contour_area: float
    perimeter: float

    def to_dict(self) -> dict:
        data = asdict(self)
        data["confidence"] = round(self.confidence, 4)
        data["distance"] = round(self.distance, 4)
        data["contour_area"] = round(self.contour_area, 2)
        data["perimeter"] = round(self.perimeter, 2)
        return data


@dataclass(slots=True)
class ProcessedImage:
    image_name: str
    detections: list[Detection]
    annotated_image: np.ndarray
    binary_mask: np.ndarray

    def to_dict(self) -> dict:
        return {
            "image": self.image_name,
            "detections": [detection.to_dict() for detection in self.detections],
        }


@dataclass(slots=True)
class ReferenceShape:
    label: str
    contour: np.ndarray
    source_path: Path


@dataclass(slots=True)
class DetectionParams:
    threshold_value: int = 140
    min_area: int = 400
    morph_kernel_size: int = 3
    match_threshold: float = 0.18


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and is_supported_image(path):
            yield path


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    return image


def normalize_label(path: Path) -> str:
    stem = path.stem.lower().replace(" ", "_")
    if stem == "rectangle":
        return "rectangle_outline"
    return stem


def preprocess_image(
    image: np.ndarray,
    *,
    threshold_value: int,
    morph_kernel_size: int,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY_INV,
    )
    kernel_size = max(1, morph_kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def find_external_contours(binary_mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours


def extract_primary_contour(
    image: np.ndarray,
    *,
    threshold_value: int = 200,
    morph_kernel_size: int = 3,
    min_area: int = 50,
) -> np.ndarray:
    binary = preprocess_image(
        image,
        threshold_value=threshold_value,
        morph_kernel_size=morph_kernel_size,
    )
    contours = find_external_contours(binary)
    valid = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not valid:
        raise ValueError("No se encontro un contorno principal valido en la imagen de referencia.")
    return max(valid, key=cv2.contourArea)


def load_reference_shapes(input_dir: Path) -> list[ReferenceShape]:
    if not input_dir.exists():
        raise FileNotFoundError(f"La carpeta de referencias no existe: {input_dir}")

    references: list[ReferenceShape] = []
    for path in iter_images(input_dir):
        image = load_image(path)
        contour = extract_primary_contour(image)
        references.append(
            ReferenceShape(
                label=normalize_label(path),
                contour=contour,
                source_path=path,
            )
        )

    if not references:
        raise FileNotFoundError(
            f"No se encontraron imagenes de referencia en: {input_dir}"
        )
    return references


def classify_contour(
    contour: np.ndarray,
    references: list[ReferenceShape],
    *,
    match_threshold: float,
) -> tuple[str, float, float]:
    best_reference: ReferenceShape | None = None
    best_distance = float("inf")

    for reference in references:
        distance = cv2.matchShapes(
            contour,
            reference.contour,
            cv2.CONTOURS_MATCH_I1,
            0.0,
        )
        if distance < best_distance:
            best_distance = distance
            best_reference = reference

    if best_reference is None or best_distance > match_threshold:
        confidence = max(0.0, 1.0 - best_distance / max(match_threshold, 1e-6))
        return "unknown", confidence, best_distance

    confidence = max(0.0, 1.0 - best_distance / max(match_threshold, 1e-6))
    return best_reference.label, confidence, best_distance


def detect_shapes(
    image: np.ndarray,
    references: list[ReferenceShape],
    params: DetectionParams,
) -> tuple[list[Detection], np.ndarray]:
    binary = preprocess_image(
        image,
        threshold_value=params.threshold_value,
        morph_kernel_size=params.morph_kernel_size,
    )
    contours = find_external_contours(binary)
    detections: list[Detection] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < params.min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        label, confidence, distance = classify_contour(
            contour,
            references,
            match_threshold=params.match_threshold,
        )
        x, y, width, height = cv2.boundingRect(contour)
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                distance=distance,
                bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
                contour_area=area,
                perimeter=perimeter,
            )
        )

    detections.sort(key=lambda item: (item.bounding_box.y, item.bounding_box.x))
    return detections, binary


def process_image(
    image_path: Path,
    references: list[ReferenceShape],
    params: DetectionParams,
) -> ProcessedImage:
    from .drawing import draw_detections

    image = load_image(image_path)
    detections, binary = detect_shapes(image, references, params)
    annotated = draw_detections(image, detections)
    return ProcessedImage(
        image_name=image_path.name,
        detections=detections,
        annotated_image=annotated,
        binary_mask=binary,
    )
