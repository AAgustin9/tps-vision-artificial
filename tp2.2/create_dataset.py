"""
Generador de dataset sintetico (tp2.2).

Dibuja formas con OpenCV bajo diferentes posiciones, escalas y
rotaciones, extrae sus invariantes de Hu y guarda el resultado en
data/hu_moments.csv.

Ejecucion:
    python create_dataset.py
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np

from labels import LABELS


IMG_SIZE = 500
N_SAMPLES_PER_CLASS = 40
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def hu_from_contour(contour: np.ndarray) -> list[float]:
    # Computes the 7 Hu moment invariants for a contour and returns them as a flat list.
    m = cv2.moments(contour)
    hu = cv2.HuMoments(m).flatten()
    return hu.tolist()


def extract_largest_contour(binary: np.ndarray) -> np.ndarray | None:
    # Finds all external contours in a binary mask and returns the largest one by area,
    # or None if no contours are found.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def random_position(margin: int) -> tuple[int, int]:
    # Returns a random (x, y) position within the image bounds, keeping at least margin pixels from the edges.
    lo, hi = margin, IMG_SIZE - margin
    return random.randint(lo, hi), random.randint(lo, hi)


def generate_circle_samples(n: int) -> list[list[float]]:
    # Generates n samples for the circle class by drawing filled circles at random
    # positions and radii on a blank image and computing their Hu moments.
    samples = []
    while len(samples) < n:
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        r = random.randint(40, 110)
        cx, cy = random_position(r + 5)
        cv2.circle(img, (cx, cy), r, 255, -1)
        contour = extract_largest_contour(img)
        if contour is not None:
            samples.append(hu_from_contour(contour))
    return samples


def generate_rectangle_samples(n: int) -> list[list[float]]:
    # Generates n samples for the rectangle class by drawing filled rectangles with
    # random size, position, and rotation angle, then computing their Hu moments.
    samples = []
    while len(samples) < n:
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        w = random.randint(60, 180)
        h = random.randint(40, 130)
        cx, cy = random_position(max(w, h) // 2 + 10)
        x1, y1 = cx - w // 2, cy - h // 2
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), 255, -1)
        # random rotation
        angle = random.uniform(0, 180)
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
        img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
        contour = extract_largest_contour(img)
        if contour is not None and cv2.contourArea(contour) > 200:
            samples.append(hu_from_contour(contour))
    return samples


def draw_star(
    img: np.ndarray,
    cx: int,
    cy: int,
    r_outer: int,
    n_points: int = 5,
    angle_offset: float = 0.0,
) -> None:
    # Draws a filled n-pointed star centered at (cx, cy) with the given outer radius.
    # The inner radius is 40% of the outer. angle_offset rotates the whole star.
    r_inner = r_outer * 0.4
    pts = []
    for i in range(2 * n_points):
        angle = math.pi / n_points * i + angle_offset - math.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 255)


def generate_star_samples(n: int) -> list[list[float]]:
    # Generates n samples for the star class by drawing 5-pointed stars at random
    # positions, sizes, and rotation angles, then computing their Hu moments.
    samples = []
    while len(samples) < n:
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        r = random.randint(50, 120)
        cx, cy = random_position(r + 10)
        angle_offset = random.uniform(0, 2 * math.pi)
        draw_star(img, cx, cy, r, n_points=5, angle_offset=angle_offset)
        contour = extract_largest_contour(img)
        if contour is not None and cv2.contourArea(contour) > 200:
            samples.append(hu_from_contour(contour))
    return samples


def main() -> None:
    # Generates synthetic Hu moment samples for each shape class, shuffles them,
    # and writes the full dataset to data/hu_moments.csv.
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "hu_moments.csv"

    generators = {
        1: generate_circle_samples,
        2: generate_rectangle_samples,
        3: generate_star_samples,
    }

    rows: list[list] = []
    for label_id, generator in generators.items():
        print(f"Generando muestras para '{LABELS[label_id]}'...")
        for hu in generator(N_SAMPLES_PER_CLASS):
            rows.append(hu + [label_id])

    random.shuffle(rows)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7", "label"])
        writer.writerows(rows)

    print(f"\nDataset guardado: {csv_path} ({len(rows)} muestras total)")
    for label_id, name in LABELS.items():
        count = sum(1 for r in rows if r[-1] == label_id)
        print(f"  {name} (label={label_id}): {count} muestras")


if __name__ == "__main__":
    main()
