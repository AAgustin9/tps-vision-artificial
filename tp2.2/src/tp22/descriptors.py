from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FEATURE_COLUMNS = [f"hu_{index}" for index in range(1, 8)]


@dataclass(slots=True)
class DescriptorRecord:
    label: str
    features: np.ndarray
    source: str

    def to_row(self) -> dict[str, str | float]:
        row: dict[str, str | float] = {"label": self.label, "source": self.source}
        for column, value in zip(FEATURE_COLUMNS, self.features.tolist(), strict=True):
            row[column] = float(value)
        return row


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


def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        5,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def extract_external_contours(binary: np.ndarray, min_area: int = 50) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) >= min_area]


def contour_to_hu_moments(contour: np.ndarray) -> np.ndarray:
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    safe = np.where(np.abs(hu) < 1e-12, 1e-12, np.abs(hu))
    signed = np.sign(hu)
    return (-signed * np.log10(safe)).astype(np.float64)


def largest_contour_descriptor(image: np.ndarray, min_area: int = 50) -> np.ndarray:
    binary = preprocess_image(image)
    contours = extract_external_contours(binary, min_area=min_area)
    if not contours:
        raise ValueError("No se encontraron contornos utiles para calcular descriptores.")
    contour = max(contours, key=cv2.contourArea)
    return contour_to_hu_moments(contour)


def collect_records_from_labeled_images(
    train_dir: Path,
    labels: Iterable[str],
    min_area: int = 50,
) -> list[DescriptorRecord]:
    records: list[DescriptorRecord] = []
    for label in labels:
        label_dir = train_dir / label
        if not label_dir.exists():
            continue
        for image_path in iter_images(label_dir):
            descriptor = largest_contour_descriptor(load_image(image_path), min_area=min_area)
            records.append(
                DescriptorRecord(
                    label=label,
                    features=descriptor,
                    source=str(image_path.relative_to(train_dir)),
                )
            )
    return records


def write_dataset_csv(records: list[DescriptorRecord], output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", *FEATURE_COLUMNS, "source"])
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    return output_csv


def read_dataset_csv(dataset_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    with dataset_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"El dataset esta vacio: {dataset_csv}")

    x = np.array(
        [
            [float(row[column]) for column in FEATURE_COLUMNS]
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.array([row["label"] for row in rows], dtype=object)
    return x, y
