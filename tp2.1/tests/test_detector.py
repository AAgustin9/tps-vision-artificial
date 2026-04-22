from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

import cv2
import numpy as np

from src.tp21.detector import (
    DetectionParams,
    detect_shapes,
    load_reference_shapes,
)


def make_canvas(width: int = 420, height: int = 420) -> np.ndarray:
    # Creates a blank white BGR image of the given dimensions, used as a drawing surface in tests.
    return np.full((height, width, 3), 255, dtype=np.uint8)


def draw_star(
    canvas: np.ndarray,
    center: tuple[int, int],
    outer_radius: int,
    inner_radius: int,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    # Draws a filled 5-pointed star on canvas, alternating between outer and inner radii
    # to produce the classic star polygon shape used in tests.
    points = []
    center_x, center_y = center
    for index in range(10):
        angle = -pi / 2 + index * (pi / 5)
        radius = outer_radius if index % 2 == 0 else inner_radius
        x = int(center_x + radius * cos(angle))
        y = int(center_y + radius * sin(angle))
        points.append((x, y))
    polygon = np.array(points, dtype=np.int32)
    cv2.fillPoly(canvas, [polygon], color)


def write_references(tmp_path: Path) -> Path:
    # Creates a temporary references directory with one circle, one rectangle outline,
    # and one star image. Returns the directory path for use in tests.
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()

    circle = make_canvas(160, 160)
    cv2.circle(circle, (80, 80), 50, (0, 0, 0), -1)
    cv2.imwrite(str(refs_dir / "circle.png"), circle)

    rectangle = make_canvas(220, 160)
    cv2.rectangle(rectangle, (30, 35), (190, 125), (0, 0, 0), 10)
    cv2.imwrite(str(refs_dir / "rectangle.png"), rectangle)

    star = make_canvas(180, 180)
    draw_star(star, center=(90, 90), outer_radius=70, inner_radius=28)
    cv2.imwrite(str(refs_dir / "star.png"), star)

    return refs_dir


def build_params() -> DetectionParams:
    # Returns a DetectionParams instance tuned for the synthetic test images.
    return DetectionParams(
        threshold_value=140,
        min_area=200,
        morph_kernel_size=3,
        match_threshold=0.2,
    )


def test_load_reference_shapes_uses_file_names_as_labels(tmp_path: Path) -> None:
    # Verifies that load_reference_shapes derives labels from the image filenames
    # and applies the "rectangle" → "rectangle_outline" normalization.
    refs_dir = write_references(tmp_path)
    references = load_reference_shapes(refs_dir)
    labels = sorted(reference.label for reference in references)
    assert labels == ["circle", "rectangle_outline", "star"]


def test_detects_shapes_using_reference_matching(tmp_path: Path) -> None:
    # Draws a circle, rectangle outline, and star on a synthetic frame and verifies
    # that all three are detected and correctly labeled via matchShapes.
    refs_dir = write_references(tmp_path)
    references = load_reference_shapes(refs_dir)
    params = build_params()

    frame = make_canvas(760, 300)
    cv2.circle(frame, (110, 150), 58, (0, 0, 0), -1)
    cv2.rectangle(frame, (250, 90), (470, 210), (0, 0, 0), 10)
    draw_star(frame, center=(630, 150), outer_radius=72, inner_radius=30)

    detections, _ = detect_shapes(frame, references, params)
    labels = sorted(detection.label for detection in detections)
    assert labels == ["circle", "rectangle_outline", "star"]


def test_marks_unknown_when_distance_is_above_threshold(tmp_path: Path) -> None:
    # Verifies that a shape with no close reference match (a triangle) is labeled "unknown"
    # when the match threshold is set very low.
    refs_dir = write_references(tmp_path)
    references = load_reference_shapes(refs_dir)
    params = build_params()
    params.match_threshold = 0.02

    frame = make_canvas(280, 280)
    triangle = np.array([(140, 40), (50, 220), (230, 220)], dtype=np.int32)
    cv2.fillPoly(frame, [triangle], (0, 0, 0))

    detections, _ = detect_shapes(frame, references, params)

    assert len(detections) == 1
    assert detections[0].label == "unknown"
