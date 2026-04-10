from __future__ import annotations

from math import cos, pi, sin

import cv2
import numpy as np

from .descriptors import DescriptorRecord, contour_to_hu_moments, extract_external_contours, preprocess_image


LABELS = ("circle", "rectangle_outline", "star")


def make_canvas(size: int = 420) -> np.ndarray:
    return np.full((size, size, 3), 255, dtype=np.uint8)


def _draw_rotated_rectangle_outline(
    canvas: np.ndarray,
    center: tuple[int, int],
    width: int,
    height: int,
    angle: float,
    thickness: int,
) -> None:
    box = cv2.boxPoints(((center[0], center[1]), (width, height), angle))
    box = np.round(box).astype(np.int32)
    cv2.polylines(canvas, [box], isClosed=True, color=(0, 0, 0), thickness=thickness)


def _draw_star(
    canvas: np.ndarray,
    center: tuple[int, int],
    outer_radius: int,
    inner_radius: int,
    angle_offset: float,
) -> None:
    points: list[tuple[int, int]] = []
    center_x, center_y = center
    for index in range(10):
        angle = angle_offset - pi / 2 + index * (pi / 5)
        radius = outer_radius if index % 2 == 0 else inner_radius
        x = int(center_x + radius * cos(angle))
        y = int(center_y + radius * sin(angle))
        points.append((x, y))
    polygon = np.array(points, dtype=np.int32)
    cv2.fillPoly(canvas, [polygon], (0, 0, 0))


def generate_shape_image(label: str, rng: np.random.Generator, size: int = 420) -> np.ndarray:
    canvas = make_canvas(size=size)
    center = (
        int(rng.integers(int(size * 0.38), int(size * 0.62))),
        int(rng.integers(int(size * 0.38), int(size * 0.62))),
    )

    if label == "circle":
        radius = int(rng.integers(int(size * 0.16), int(size * 0.26)))
        cv2.circle(canvas, center, radius, (0, 0, 0), -1)
    elif label == "rectangle_outline":
        width = int(rng.integers(int(size * 0.32), int(size * 0.50)))
        height = int(rng.integers(int(size * 0.20), int(size * 0.42)))
        thickness = int(rng.integers(8, 18))
        angle = float(rng.uniform(-35.0, 35.0))
        _draw_rotated_rectangle_outline(canvas, center, width, height, angle, thickness)
    elif label == "star":
        outer_radius = int(rng.integers(int(size * 0.18), int(size * 0.28)))
        inner_radius = int(outer_radius * rng.uniform(0.34, 0.50))
        angle_offset = float(rng.uniform(-0.5, 0.5))
        _draw_star(canvas, center, outer_radius, inner_radius, angle_offset)
    else:
        raise ValueError(f"Etiqueta no soportada: {label}")

    if rng.random() < 0.4:
        sigma = float(rng.uniform(0.2, 1.2))
        canvas = cv2.GaussianBlur(canvas, (5, 5), sigma)

    if rng.random() < 0.5:
        noise = rng.normal(loc=0.0, scale=rng.uniform(2.0, 10.0), size=canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return canvas


def generate_synthetic_records(
    samples_per_class: int,
    seed: int = 42,
    min_area: int = 50,
) -> list[DescriptorRecord]:
    rng = np.random.default_rng(seed)
    records: list[DescriptorRecord] = []

    for label in LABELS:
        for index in range(samples_per_class):
            image = generate_shape_image(label, rng)
            binary = preprocess_image(image)
            contours = extract_external_contours(binary, min_area=min_area)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            records.append(
                DescriptorRecord(
                    label=label,
                    features=contour_to_hu_moments(contour),
                    source=f"synthetic/{label}_{index:03d}.png",
                )
            )

    return records
