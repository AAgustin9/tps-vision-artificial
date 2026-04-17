"""
Generador de dataset (tp2.2).

Lee imagenes de data/shapes/<label>/, aplica aumentacion
(rotaciones x escalas) y extrae invariantes de Hu con transformacion
logaritmica. Guarda el resultado en data/hu_moments.csv.

Ejecucion:
    python create_dataset.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np

from labels import LABELS


SHAPES_DIR = Path("data/shapes")
CSV_PATH = Path("data/hu_moments.csv")

ROTATION_ANGLES = (0, -20, 20)
SCALE_FACTORS = (1.0, 0.85)


def augment(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    variants = []
    for angle in ROTATION_ANGLES:
        for scale in SCALE_FACTORS:
            M = cv2.getRotationMatrix2D(center, angle, scale)
            warped = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            variants.append(warped)
    return variants


def log_hu(contour: np.ndarray) -> list[float] | None:
    m = cv2.moments(contour)
    hu = cv2.HuMoments(m).flatten()
    result = []
    for v in hu:
        if v == 0.0:
            result.append(0.0)
        else:
            result.append(-math.copysign(1.0, v) * math.log10(abs(v)))
    return result


def extract_largest_contour(binary: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def hu_from_image(img: np.ndarray) -> list[float] | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 67, 2,
    )
    binary = 255 - binary
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)
    contour = extract_largest_contour(binary)
    if contour is None or cv2.contourArea(contour) < 200:
        return None
    return log_hu(contour)


def main() -> None:
    NAME_TO_ID = {v: k for k, v in LABELS.items()}

    if not SHAPES_DIR.exists():
        raise FileNotFoundError(
            f"Directorio {SHAPES_DIR} no encontrado.\n"
            "Ejecuta primero: python generate_descriptors.py"
        )

    labels_found = sorted(
        d.name for d in SHAPES_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not labels_found:
        raise ValueError(f"No se encontraron subdirectorios en {SHAPES_DIR}.")

    rows: list[list] = []
    for label_name in labels_found:
        label_id = NAME_TO_ID.get(label_name)
        if label_id is None:
            print(f"  Advertencia: clase '{label_name}' no esta en labels.py, ignorando.")
            continue

        images = list((SHAPES_DIR / label_name).glob("*.png")) + \
                 list((SHAPES_DIR / label_name).glob("*.jpg")) + \
                 list((SHAPES_DIR / label_name).glob("*.jpeg"))
        if not images:
            print(f"  Advertencia: no hay imagenes en {SHAPES_DIR / label_name}")
            continue

        count = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            for variant in augment(img):
                hu = hu_from_image(variant)
                if hu is not None:
                    rows.append(hu + [label_id])
                    count += 1

        print(f"  {label_name} (label={label_id}): {count} muestras "
              f"({len(images)} imagenes x {len(ROTATION_ANGLES) * len(SCALE_FACTORS)} variantes)")

    if not rows:
        raise ValueError("No se generaron muestras. Verifica las imagenes en data/shapes/.")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7", "label"])
        writer.writerows(rows)

    print(f"\nDataset guardado: {CSV_PATH} ({len(rows)} muestras total)")


if __name__ == "__main__":
    main()
